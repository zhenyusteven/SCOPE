from __future__ import annotations
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Iterable, Callable, Tuple
from pathlib import Path
import re
import json

from ast_parser import ProjectParser


# ============ Node ============

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

    # 内容层
    source: Optional[str] = None      # 叶子代码
    summary: Optional[str] = None     # LLM 摘要（也可给叶子）
    meta: Dict[str, object] = field(default_factory=dict)  # file/qualname/pos 等

    def add_child(self, child: "CodeNode") -> None:
        self.children.append(child.id)
        child.parent = self.id

    def is_leaf(self) -> bool:
        return len(self.children) == 0

    def text_for_completion(
        self,
        include: str = "best",  # "best"|"both"|"summary"|"source"
        max_chars: int = 10_000
    ) -> str:
        """
        为补全/生成准备的文本片段：
          - best: 叶子优先 source，否则 summary
          - both: summary + source
          - summary: 只要摘要
          - source: 只要源码
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
        """递归转换为可序列化的 dict"""
        data = {
            "id": self.id,
            "name": self.name,
            "kind": self.kind,
            "summary": self.summary,
            "children": [],
            "meta": self.meta,
        }
        if include_source and self.source:
            data["source"] = self.source[:50] + ("..." if len(self.source) > 50 else "")

        return data

# ============ Tree ============

class CodeSemanticTree:
    def __init__(self, parser: ProjectParser, project_name: str = "Project"):
        self.parser = parser
        self.nodes: Dict[str, CodeNode] = {}
        self.root = CodeNode(id="root", name=project_name, kind="project")
        self.nodes[self.root.id] = self.root
        
    # ---- 基础增删改查 ----
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

    # ---- 构建：从 ProjectParser ----
    def build_from_parser(self, parser: "ProjectParser") -> None:
        """
        文件/类为中间节点；函数/方法为叶子；叶子带 source。
        中间节点 summary 留空，后续可由 LLM 填充。
        meta: file/qualname/kind/position 等。
        """
        for file in parser.get_all_files():
            file_id = str(file)
            # 从 parser 拿整个文件源码
            file_src = parser._sources.get(file, file.read_text(encoding="utf-8"))
            file_node = CodeNode(
                id=file_id,
                name=Path(file).name,
                kind="file",
                source=file_src,       
                meta={"file": str(file)},
            )
            self.add_child(self.root.id, file_node)

            # 预先缓存类节点，便于方法挂载
            class_ids: Dict[str, str] = {}

            for rec in parser.list_symbols(file):
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
                            "file": str(file),
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
                            "file": str(file),
                            "qualname": rec.qualname,
                            "position": (rec.position.start_line, rec.position.end_line),
                        },
                    )
                    # 方法挂到类；函数挂到文件
                    if rec.kind == "method":
                        cls_name = ".".join(rec.qualname.split(".")[:-1])
                        parent_id = class_ids.get(cls_name, file_id)
                        self.add_child(parent_id, leaf)
                    else:
                        self.add_child(file_id, leaf)

    # ---- 遍历/导航 ----
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
            # children 从左到右
            stack.extend(reversed(self.nodes[nid].children))

    # ---- 搜索（name or qualname 正则）----
    def find(self, pattern: str) -> List[str]:
        r = re.compile(pattern)
        return [nid for nid, n in self.nodes.items() if r.search(n.name)]

    # ---- 相关性打分（可替换）----
    @staticmethod
    def _default_score(query: str, node: CodeNode) -> float:
        """
        轻量关键字重叠得分：在 name + summary + (叶子)source 中匹配 query 的 token。
        """
        q = [t for t in re.split(r"\W+", query.lower()) if t]
        hay = " ".join([
            node.name or "",
            (node.summary or ""),
            (node.source or "")[:4000]  # 防止极长文本影响速度
        ]).lower()
        if not q or not hay:
            return 0.0
        hit = sum(hay.count(t) for t in q)
        # 适当提升叶子优先级
        leaf_bonus = 0.5 if node.is_leaf() else 0.0
        return hit + leaf_bonus

    @staticmethod
    def _approx_tokens(text: str) -> int:
        """非常粗的 token 估算。按 4 chars ≈ 1 token。"""
        if not text:
            return 0
        return max(1, len(text) // 4)

    # ---- 上下文收集：从 root 出发按相关性下钻，受 token 预算限制 ----
    def collect_context(
        self,
        query: str,
        token_budget: int = 1200,
        include_order: Tuple[str, ...] = ("ancestors", "self", "siblings", "children"),
        include_mode: str = "best",    # 节点文本模式：best|both|summary|source
        score_fn: Optional[Callable[[str, CodeNode], float]] = None,
    ) -> Dict[str, str]:
        """
        返回 {node_id: text}，用于放入 LLM prompt。
        策略：
          1) 计算每个节点的相关度得分
          2) 优先遍历高分分支（best-first）
          3) 对每个落点，按 include_order 依次尝试加入文本，直到 token_budget 用尽
        """
        score_fn = score_fn or self._default_score

        # 预计算得分
        scores: Dict[str, float] = {nid: score_fn(query, n) for nid, n in self.nodes.items()}

        # best-first：从 root 出发，优先走高分 children
        out: Dict[str, str] = {}
        used_tokens = 0

        #在不超过 token 限制的前提下，逐个添加最相关的节点内容
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

        # 按得分排序 children
        def sorted_children(nid: str) -> List[str]:
            ch = self.nodes[nid].children
            return sorted(ch, key=lambda cid: scores[cid], reverse=True)

        # 遍历
        frontier = ["root"]
        visited = set()
        while frontier and used_tokens < token_budget:
            nid = frontier.pop(0)
            if nid in visited:
                continue
            visited.add(nid)

            # 当前落点，按策略收集
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

            # 继续向下扩展 frontier（按高分 child）
            frontier[:0] = sorted_children(nid)

        return out
    
    def to_json(self, include_source: bool = True, indent: int = 2) -> str:
        """导出整棵语义树为 JSON 字符串"""
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

    def edit_symbol(
        self,
        file_path: str | Path,
        symbol_path: str,
        relative_line: int,
        new_code: str,
        mode: str = "insert",
    ):
        """编辑树中一个函数/方法节点，并同步 parser 与树"""
        file = str(Path(file_path))
        record, new_src = self.parser.edit_symbol(file, symbol_path, relative_line, new_code, mode)

        # 更新 parser 缓存（parser 自己已经做了）
        # 更新 file 层节点源码
        file_node = self.nodes.get(file)
        if file_node:
            file_node.source = new_src

        # 更新目标函数节点的源码
        target_id = f"{file}::{symbol_path}"
        if target_id in self.nodes:
            func_node = self.nodes[target_id]
            func_node.source = self.parser.get_source_with_context(record)
        else:
            # 若树中没该节点（新增函数），则重建该文件的分支
            self._rebuild_file_branch(file)

    def _rebuild_file_branch(self, file_path: str):
        """重建某个文件的节点分支"""
        file = Path(file_path)
        file_id = str(file)
        # 先清理旧的 file 子树
        to_delete = [nid for nid in self.nodes if nid.startswith(file_id + "::")]
        for nid in to_delete:
            self.nodes.pop(nid, None)

        # 重新解析并挂载子节点
        recs = self.parser.list_symbols(file)
        class_nodes = {}

        for rec in recs:
            node_id = f"{file_id}::{rec.qualname}"
            if rec.kind == "class":
                cls_node = CodeNode(
                    id=node_id,
                    name=rec.qualname,
                    kind="class",
                    source=self.parser.get_source(rec),
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

    # ---- 展示 ----
    def display(self, node_id: str = "root", indent: int = 0) -> None:
        node = self.nodes[node_id]
        prefix = "  " * indent
        label = ''
        #label = node.summary[:60] + "..." if (node.summary and len(node.summary) > 60) else (node.summary or "")
        print(f"{prefix}- {node.kind}: {node.name}  {('['+label+']') if label else ''}")
        for cid in node.children:
            self.display(cid, indent + 1)
