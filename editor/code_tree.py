from __future__ import annotations
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Iterable, Callable, Tuple
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
import re
import json
import os
try:
    from .ast_parser import ProjectParser
except ImportError:
    from ast_parser import ProjectParser
import time
from openai import OpenAI
import textwrap
from tqdm import tqdm

def llm_summarize(text: str) -> str:
    api_key = os.environ.get("OPENAI_API_KEY")
    if api_key:
        try:
            client = OpenAI(api_key=api_key) 
            prompt = textwrap.dedent(f'''
            You are a static-code summarization assistant. 
            Given code or child summaries, produce a *compact, machine-readable* JSON object describing the node.
            Required fields:
            - "role": short label (e.g., "utility fn", "stateful class", "config file", "API wrapper")
            - "purpose": 1-2 sentences on what this node does
            - "deps": "internal": referenced project functions/classes/modules, "external": imported libraries
            - "contracts": key assumptions or pre/post conditions
            - "effects": side effects (I/O, logs, state)
            - "errors": edge cases or exceptions
            - "risks": things an edit must not break
            - "links": any related nodes
            - "sum": 2-3 sentence natural summary
            - "conf": confidence (0–1)
            Be extremely concise and strictly output JSON only.
            {text}
            ''')
            
            resp = client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[{"role": "user", "content": prompt}],
                max_tokens=400,
                temperature=0.2,
            )
            return resp.choices[0].message.content.strip()
        except Exception as e:
            print(f"[WARN] OpenAI summarization failed: {e}")
    else:
        print("no openai key found")
        return "NO_API_KEY"
    
@dataclass
class CodeNode:
    """
    Code tree node definition.
    """
    id: str
    name: str
    kind: str
    parent: Optional[str] = None
    children: List[str] = field(default_factory=list)

    # Content layer
    source: Optional[str] = None      # Leaf code
    summary: Optional[str] = None     # LLM summary (may exist on leaves)
    meta: Dict[str, object] = field(default_factory=dict)  # file/qualname/pos etc

    def add_child(self, child: "CodeNode") -> None:
        self.children.append(child.id)
        child.parent = self.id

    def is_leaf(self) -> bool:
        return len(self.children) == 0

    def text_for_completion(
        self,
        include: str = "best",  # "best"|"both"|"summary"|"source"
        max_chars: int = 10000
    ) -> str:
        """
        Prepare text snippets for completion/generation:
          - best: prefer leaf source; otherwise summary
          - both: summary + source
          - summary: summary only
          - source: source only
        """
        blocks: List[str] = []
        if include == "best":
            if self.source:
                blocks.append(self.source)
            elif self.summary:
                blocks.append(self.summary)
        elif include == "both":
            if self.summary:
                blocks.append(self.summary)
            if self.source:
                blocks.append(self.source)
        elif include == "summary":
            if self.summary:
                blocks.append(self.summary)
        elif include == "source":
            if self.source:
                blocks.append(self.source)

        out = "\n".join(blocks).strip()
        return out[:max_chars]
     
    def to_dict(self, include_source: bool = True) -> Dict:
        """Recursively convert to a serializable dict"""
        data = {
            "id": self.id,
            "name": self.name,
            "kind": self.kind,
            "summary": self.summary,
            "children": [],
            "meta": self.meta,
        }
        if include_source and self.source:
            data["source"] = self.source #[:50] + ("..." if len(self.source) > 50 else "")

        return data

# ============ Tree ============

class CodeSemanticTree:
    def __init__(self, parser: ProjectParser = None, tree_path: str = None, project_name: str = "Project"):
        assert not (parser is None and tree_path is None), "Either parser or tree_path must be provided"
        if parser is not None:
            self.parser = parser
            self.nodes: Dict[str, CodeNode] = {}
            self.root = CodeNode(id="root", name=project_name, kind="project")
            self.nodes[self.root.id] = self.root
        else:
            loaded = self.create_from_json(tree_path)
            self.parser = None
            self.nodes = loaded.nodes
            self.root = loaded.root
        
        
    # ---- Basic CRUD helpers ----
    def add_node(self, node: CodeNode) -> None:
        self.nodes[node.id] = node

    def add_child(self, parent_id: str, child: CodeNode) -> None:
        parent = self.nodes[parent_id]
        parent.add_child(child)
        self.add_node(child)

    def get(self, node_id: str) -> CodeNode:
        return self.nodes[node_id]

    def set_summary(self, node_id: str, summary: str) -> None:
        self.nodes[node_id].summary = summary

    def set_source(self, node_id: str, source: str) -> None:
        self.nodes[node_id].source = source

    def _normalize_file_path(self, file_path: str | Path) -> Tuple[str, Path]:
        """
        Normalize incoming paths so IDs in the tree remain relative to the parser root.
        """
        abs_path = self.parser._abs(file_path)
        try:
            rel_path = abs_path.relative_to(self.parser.root)
            return rel_path.as_posix(), abs_path
        except ValueError:
            # File is outside the project root; fall back to absolute id.
            return abs_path.as_posix(), abs_path
 
    
    # ---- Build: from ProjectParser ----
    def build_from_parser(self, parser: "ProjectParser") -> None:
        """
        Files/classes become intermediate nodes; functions/methods are leaves and carry source.
        Intermediate node summaries stay empty and can be filled by an LLM later.
        meta: file/qualname/kind/position, etc.
        """     
        for file in parser.get_all_files():
            file_id, abs_file = self._normalize_file_path(file)
            # Pull entire file source from the parser
            file_src = parser._sources.get(abs_file, abs_file.read_text(encoding="utf-8"))
            file_node = CodeNode(
                id=file_id,
                name=Path(abs_file).name,
                kind="file",
                source=file_src,       
                meta={"file": file_id},
            )
            self.add_child(self.root.id, file_node)

            # Pre-cache class nodes so methods can attach
            class_ids: Dict[str, str] = {}

            for rec in parser.list_symbols(abs_file):
                node_id = f"{file_id}::{rec.qualname}"
                src = parser.get_source_with_context(rec, before=0, after=0)
                if rec.kind == "class":
                    cls = CodeNode(
                        id=node_id,
                        name=rec.qualname,
                        kind="class",
                        source=src,
                        summary=None,
                        meta={
                            "file": file_id,
                            "qualname": rec.qualname,
                            "position": (rec.position.start_line, rec.position.end_line),
                        },
                    )
                    self.add_child(file_id, cls)
                    class_ids[rec.qualname] = node_id

                elif rec.kind in ("function", "method"):
                    src = parser.get_source_with_context(rec, before=0, after=0)
                    leaf = CodeNode(
                        id=node_id,
                        name=rec.qualname,
                        kind=rec.kind,
                        source=src,
                        meta={
                            "file": file_id,
                            "qualname": rec.qualname,
                            "position": (rec.position.start_line, rec.position.end_line),
                        },
                    )
                    # Methods attach to classes; functions attach to files
                    if rec.kind == "method":
                        cls_name = ".".join(rec.qualname.split(".")[:-1])
                        parent_id = class_ids.get(cls_name, file_id)
                        self.add_child(parent_id, leaf)
                    else:
                        self.add_child(file_id, leaf)


    def build_from_parser_with_folder(self, parser: "ProjectParser") -> None:
        """
        Files/classes are intermediate nodes; functions/methods are leaves carrying source.
        Create a folder node for every directory encountered.
        """
        root_dir = self.parser.root
        for file in parser.get_all_files():
            file_id, abs_file = self._normalize_file_path(file)
            rel_parts = abs_file.relative_to(root_dir).parts  # split relative path
            parent_id = self.root.id
            path_accum = []

            # === Step 1: ensure each folder node exists ===
            for part in rel_parts[:-1]:  # ignore file name; only process directories
                path_accum.append(part)
                folder_path = Path(*path_accum).as_posix()
                if folder_path not in self.nodes:
                    folder_node = CodeNode(
                        id=folder_path,
                        name=part,
                        kind="folder",
                        source=None,
                        summary=None,
                        meta={"path": folder_path},
                    )
                    self.add_child(parent_id, folder_node)
                parent_id = folder_path

            # === Step 2: add file node ===
            file_src = parser._sources.get(abs_file, abs_file.read_text(encoding="utf-8"))
            file_node = CodeNode(
                id=file_id,
                name=Path(abs_file).name,
                kind="file",
                source=file_src,
                meta={"file": file_id},
            )
            self.add_child(parent_id, file_node)

            # === Step 3: add class and function nodes ===
            class_ids: Dict[str, str] = {}
            for rec in parser.list_symbols(abs_file):
                node_id = f"{file_id}::{rec.qualname}"
                src = parser.get_source_with_context(rec, before=0, after=0)

                if rec.kind == "class":
                    cls = CodeNode(
                        id=node_id,
                        name=rec.qualname,
                        kind="class",
                        source=src,
                        meta={
                            "file": file_id,
                            "qualname": rec.qualname,
                            "position": (rec.position.start_line, rec.position.end_line),
                        },
                    )
                    self.add_child(file_id, cls)
                    class_ids[rec.qualname] = node_id

                elif rec.kind in ("function", "method"):
                    leaf = CodeNode(
                        id=node_id,
                        name=rec.qualname,
                        kind=rec.kind,
                        source=src,
                        meta={
                            "file": file_id,
                            "qualname": rec.qualname,
                            "position": (rec.position.start_line, rec.position.end_line),
                        },
                    )
                    if rec.kind == "method":
                        cls_name = ".".join(rec.qualname.split(".")[:-1])
                        parent_id_ = class_ids.get(cls_name, file_id)
                    else:
                        parent_id_ = file_id
                    self.add_child(parent_id_, leaf)

    # ---- Traversal/navigation ----
    def ancestors(self, node_id: str) -> List[str]:
        out: List[str] = []
        cur = self.nodes[node_id].parent
        while cur:
            out.append(cur)
            cur = self.nodes[cur].parent
        return out[::-1]

    def path_to(self, node_id: str) -> List[str]:
        return self.ancestors(node_id) + [node_id]

    def iter_dfs(self, start_id: str = "root") -> Iterable[str]:
        stack = [start_id]
        while stack:
            nid = stack.pop()
            yield nid
            # children ordered left-to-right
            stack.extend(reversed(self.nodes[nid].children))

    # ---- Search (name or qualname regex) ----
    def find(self, pattern: str) -> List[str]:
        r = re.compile(pattern)
        return [nid for nid, n in self.nodes.items() if r.search(n.name)]

    # ---- Relevance scoring (swappable) ----
    @staticmethod
    def _default_score(query: str, node: CodeNode) -> float:
        # place holder
        return 1.0 if query.lower() in node.name.lower() else 0.0

    @staticmethod
    def _approx_tokens(text: str) -> int:
        """Very rough token estimate. Roughly 4 chars ~ 1 token."""
        if not text:
            return 0
        return max(1, len(text) // 4)

    # ---- Context collection: descend from root by relevance under a token budget ----
    def collect_context(
        self,
        query: str,
        token_budget: int = 1200,
        include_order: Tuple[str, ...] = ("ancestors", "self", "siblings", "children"),
        include_mode: str = "best",    # Node text mode: best|both|summary|source
        score_fn: Optional[Callable[[str, CodeNode], float]] = None,
    ) -> Dict[str, str]:
        """
        Return {node_id: text} for use in an LLM prompt.
        Strategy:
          1) compute a relevance score per node
          2) traverse high-score branches first (best-first)
          3) at each landing node, add text per include_order until token_budget is used up
        """
        ## TODO: refine the context-collection/search strategy.
        score_fn = score_fn or self._default_score

        # Pre-compute scores
        scores: Dict[str, float] = {nid: score_fn(query, n) for nid, n in self.nodes.items()}

        # Best-first: start from root, prefer children with higher scores
        out: Dict[str, str] = {}
        used_tokens = 0

        # Add the most relevant node content while staying within the token budget
        def try_add(nid: str) -> None:

            nonlocal used_tokens
            if nid in out:
                return
            node = self.nodes[nid]
            text = node.text_for_completion(include=include_mode, max_chars=32_000)
            if not text:
                return
            need = self._approx_tokens(text)
            if used_tokens + need <= token_budget:
                out[nid] = text
                used_tokens += need

        # Sort children by score
        def sorted_children(nid: str) -> List[str]:
            ch = self.nodes[nid].children
            return sorted(ch, key=lambda cid: scores[cid], reverse=True)

        # Traverse
        frontier = ["root"]
        visited = set()
        while frontier and used_tokens < token_budget:
            nid = frontier.pop(0)
            if nid in visited:
                continue
            visited.add(nid)

            # Collect text for the current node according to the strategy
            if "ancestors" in include_order:
                for aid in self.ancestors(nid):
                    try_add(aid)

            if "self" in include_order:
                try_add(nid)

            if "siblings" in include_order:
                p = self.nodes[nid].parent
                if p:
                    for sib in self.nodes[p].children:
                        if sib != nid:
                            try_add(sib)

            if "children" in include_order:
                for cid in sorted_children(nid):
                    try_add(cid)

            # Continue expanding the frontier (prioritizing higher-score children)
            frontier[:0] = sorted_children(nid)

        return out
    
    def generate_summary(
        self,
        summarize_fn: Callable[[str], str] = llm_summarize,
        include_source_in_leaf: bool = True,
        max_workers: int = 10,
    ) -> None:
        """
        Recursively generate a summary for each node from leaves up to the root.
        summarize_fn: function (text:str) -> summary:str
        """
        ## TODO: design a smarter summary generation strategy/logic
        # Get a topological order of all nodes (leaves first)
        order = list(self.iter_dfs("root"))[::-1]
        work_nodes = [nid for nid in order if not self.nodes[nid].is_leaf()]
        if not work_nodes:
            return

        # Calculate height for better parallelism
        height: Dict[str, int] = {}
        for nid in order:
            node = self.nodes[nid]
            if node.is_leaf():
                height[nid] = 0
            else:
                child_heights = [height.get(cid, 0) for cid in node.children]
                height[nid] = 1 + (max(child_heights) if child_heights else 0)

        positions = {nid: idx for idx, nid in enumerate(order)}
        levels: Dict[int, List[str]] = defaultdict(list)
        for nid in work_nodes:
            levels[height[nid]].append(nid)
        for level_nodes in levels.values():
            level_nodes.sort(key=lambda nid: positions[nid])

        def _gather_child_text(node: CodeNode) -> str:
            child_summaries: List[str] = []
            inner_mode = "source" if include_source_in_leaf else "best"
            for cid in node.children:
                child = self.nodes[cid]
                if child.summary:
                    child_summaries.append(f"{child.name}: {child.summary}")
                else:
                    child_summaries.append(
                        f"{child.name}: {child.text_for_completion(include=inner_mode, max_chars=8000)}"
                    )
            return "\n".join(child_summaries)

        def _execute(nid: str, text: str) -> Tuple[str, Optional[str], Optional[Exception]]:
            try:
                return nid, summarize_fn(text), None
            except Exception as exc:
                return nid, None, exc

        pbar = tqdm(total=len(work_nodes))
        try:
            with ThreadPoolExecutor(max_workers=max_workers or 1) as executor:
                for depth in sorted(levels.keys()):
                    futures = []
                    for nid in levels[depth]:
                        node = self.nodes[nid]
                        text = _gather_child_text(node)
                        if not text.strip():
                            pbar.update(1)
                            continue
                        futures.append(executor.submit(_execute, nid, text))

                    for fut in as_completed(futures):
                        nid, summary, error = fut.result()
                        node = self.nodes[nid]
                        if error is None and summary is not None:
                            node.summary = summary
                            print("Summary", node.summary)
                        else:
                            print(f"[WARN] Summary failed for {node.name}: {error}")
                        pbar.update(1)
        finally:
            pbar.close()



    def to_json(self, include_source: bool = True, indent: int = 2) -> str:
        """Export the entire semantic tree as a JSON string"""
        if not self.root:
            raise ValueError("Tree has no root node")

        def build_subtree(node: CodeNode) -> Dict:
            data = node.to_dict(include_source=include_source)
            data["children"] = [
                build_subtree(self.nodes[cid]) for cid in node.children
            ]
            return data

        root_dict = build_subtree(self.root)
        return json.dumps(root_dict, indent=indent, ensure_ascii=False)
    
    def save_json(self, path: str, include_source: bool = True):
        json_str = self.to_json(include_source=include_source)
        with open(path, "w", encoding="utf-8") as f:
            f.write(json_str)
        print(f"Tree saved to {path}")

    @classmethod
    def create_from_dict(cls, data: Dict[str, Any]) -> "CodeSemanticTree":
        """Create a CodeSemanticTree from a dict"""

        def build_subtree(node_dict: Dict[str, Any], tree: "CodeSemanticTree") -> CodeNode:
            node = CodeNode(
                id=node_dict["id"],
                name=node_dict.get("name", ""),
                kind=node_dict.get("kind", ""),
                summary=node_dict.get("summary"),
                source=node_dict.get("source"),
                meta=node_dict.get("meta") or {},
            )
            tree.nodes[node.id] = node

            for child_dict in node_dict.get("children", []):
                child_node = build_subtree(child_dict, tree)
                node.add_child(child_node)

            return node

        tree = cls.__new__(cls)
        tree.parser = None
        tree.nodes = {}
        tree.root = build_subtree(data, tree)
        if tree.root.id != "root":
            tree.nodes["root"] = tree.root
        return tree

    @classmethod
    def create_from_json(cls, json_path: str) -> "CodeSemanticTree":
        """Create a CodeSemanticTree from a JSON file"""
        with open(json_path, "r", encoding="utf-8") as f:
            json_str = f.read()
        data = json.loads(json_str)
        return cls.create_from_dict(data)

    def edit_symbol(
        self,
        file_path: str | Path,
        symbol_path: str,
        relative_line: int,
        new_code: str,
        mode: str = "insert",
    ):
        """Edit a function/method node in the tree and keep the parser/tree in sync"""
        file_id, abs_file = self._normalize_file_path(file_path)
        record, new_src = self.parser.edit_symbol(abs_file, symbol_path, relative_line, new_code, mode)

        # Parser cache already refreshed; now refresh the file-level node source
        file_node = self.nodes.get(file_id)
        if file_node:
            file_node.source = new_src

        # Refresh the target function node source
        target_id = f"{file_id}::{symbol_path}"
        if target_id in self.nodes:
            func_node = self.nodes[target_id]
            func_node.source = self.parser.get_source_with_context(record)
        else:
            # If the node is missing (new function), rebuild this file branch
            self._rebuild_file_branch(abs_file)

    def _rebuild_file_branch(self, file_path: str | Path):
        """Rebuild the branch of nodes for a given file"""
        file_id, abs_file = self._normalize_file_path(file_path)
        # Clear the old file subtree first
        to_delete = [nid for nid in self.nodes if nid.startswith(file_id + "::")]
        for nid in to_delete:
            self.nodes.pop(nid, None)

        # Re-parse and attach child nodes
        recs = self.parser.list_symbols(abs_file)
        class_nodes = {}

        for rec in recs:
            node_id = f"{file_id}::{rec.qualname}"
            if rec.kind == "class":
                cls_node = CodeNode(
                    id=node_id,
                    name=rec.qualname,
                    kind="class",
                    source=self.parser.get_source(rec),
                    meta={
                        "file": file_id,
                        "qualname": rec.qualname,
                        "position": (rec.position.start_line, rec.position.end_line),
                    },
                )
                self.add_child(file_id, cls_node)
                class_nodes[rec.qualname] = cls_node
            elif rec.kind in ("function", "method"):
                src = self.parser.get_source_with_context(rec, before=0, after=0)
                func_node = CodeNode(
                    id=node_id,
                    name=rec.qualname,
                    kind=rec.kind,
                    source=src,
                    meta={
                        "file": file_id,
                        "qualname": rec.qualname,
                        "position": (rec.position.start_line, rec.position.end_line),
                    },
                )
                if rec.kind == "method":
                    cls_name = ".".join(rec.qualname.split(".")[:-1])
                    parent_id = f"{file_id}::{cls_name}"
                    if parent_id in self.nodes:
                        self.add_child(parent_id, func_node)
                    else:
                        self.add_child(file_id, func_node)
                else:
                    self.add_child(file_id, func_node)

    # ---- Display ----
    def display(self, node_id: str = "root", indent: int = 0) -> None:
        node = self.nodes[node_id]
        prefix = "  " * indent
        label = ''
        #label = node.summary[:60] + "..." if (node.summary and len(node.summary) > 60) else (node.summary or "")
        print(f"{prefix}- {node.kind}: {node.name}  {('['+label+']') if label else ''}")
        for cid in node.children:
            self.display(cid, indent + 1)
