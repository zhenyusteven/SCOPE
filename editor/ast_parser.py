from __future__ import annotations
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import libcst as cst
from libcst import metadata as cst_meta


@dataclass(frozen=True)
class Position:
    start: Tuple[int, int]
    end: Tuple[int, int]

    @property
    def start_line(self) -> int:
        return self.start[0]

    @property
    def end_line(self) -> int:
        return self.end[0]


@dataclass(frozen=True)
class SymbolRecord:
    file: Path
    qualname: str
    kind: str  # "function" | "method" | "class"
    position: Position


class _SymbolCollector(cst.CSTVisitor):
    """Collect top-level functions, classes, and methods with precise positions.

    Records fully-qualified names using dotted paths like:
      - foo
      - ClassA
      - ClassA.method_b
      - ClassA.InnerClass.method_c
    """

    METADATA_DEPENDENCIES = (cst_meta.PositionProvider,)

    def __init__(self, file: Path) -> None:
        super().__init__()
        self.file = file
        self.records: Dict[str, SymbolRecord] = {}
        self._class_stack: List[str] = []

    # --- helpers ---
    def _code_range(self, node: cst.CSTNode) -> Position:
        rng = self.get_metadata(cst_meta.PositionProvider, node)
        # CodeRange has .start.line, .start.column (0-based columns in libcst)
        start = (rng.start.line, rng.start.column)
        end = (rng.end.line, rng.end.column)
        return Position(start=start, end=end)

    def _push(self, name: str) -> None:
        self._class_stack.append(name)

    def _pop(self) -> None:
        if self._class_stack:
            self._class_stack.pop()

    def _qualname(self, name: str) -> str:
        if self._class_stack:
            return ".".join(self._class_stack + [name])
        return name

    # --- visit methods ---
    def visit_ClassDef(self, node: cst.ClassDef) -> None:
        qn = self._qualname(node.name.value)
        self.records[qn] = SymbolRecord(
            file=self.file,
            qualname=qn,
            kind="class",
            position=self._code_range(node),
        )
        self._push(node.name.value)

    def leave_ClassDef(self, node: cst.ClassDef) -> None:
        self._pop()

    def visit_FunctionDef(self, node: cst.FunctionDef) -> None:
        kind = "method" if self._class_stack else "function"
        qn = self._qualname(node.name.value)
        self.records[qn] = SymbolRecord(
            file=self.file,
            qualname=qn,
            kind=kind,
            position=self._code_range(node),
        )


class ProjectParser:
    """Parse a Python project (folder) with libcst and build a symbol index.

    Features:
      - Recursively parse *.py files
      - Index classes, functions, and methods with dotted paths
      - Retrieve exact source for a symbol and contextual slices

    Example
    -------
    >>> parser = ProjectParser("/path/to/project")
    >>> parser.build_index()
    >>> rec = parser.resolve("pkg/module.py", "ClassA.method_b")
    >>> print(parser.get_source(rec))
    """

    def __init__(self, root: Path | str) -> None:
        # Normalize so relative paths behave the same as absolute
        self.root = Path(root).expanduser().resolve()
        if not self.root.exists():
            raise FileNotFoundError(self.root)
        self._index: Dict[Path, Dict[str, SymbolRecord]] = {}
        self._wrappers: Dict[Path, cst_meta.MetadataWrapper] = {}
        self._modules: Dict[Path, cst.Module] = {}
        self._sources: Dict[Path, str] = {}

    # --- public API ---
    @property
    def index(self) -> Dict[Path, Dict[str, SymbolRecord]]:
        return self._index

    def build_index(self, *, include_tests: bool = True) -> None:
        for file in self._iter_py_files():
            try:
                src = file.read_text(encoding="utf-8")
            except Exception:
                continue  # skip unreadable files
            try:
                mod = cst.parse_module(src)
                wrapper = cst_meta.MetadataWrapper(mod)
                collector = _SymbolCollector(file)
                wrapper.visit(collector)
            except Exception:
                # Syntax errors or parser failures should not break the whole walk
                continue

            self._sources[file] = src
            self._modules[file] = mod
            self._wrappers[file] = wrapper
            self._index[file] = collector.records

        # Optionally filter out common test paths
        if not include_tests:
            self._index = {
                f: recs
                for f, recs in self._index.items()
                if not ("tests" in f.parts or f.name.startswith("test_"))
            }

    def resolve(self, file_path: str | Path, symbol_path: str) -> SymbolRecord:
        file = self._abs(file_path)
        if file not in self._index:
            raise KeyError(f"File not indexed: {file}")
        recs = self._index[file]
        if symbol_path not in recs:
            # Try a fuzzy fallback: allow leading module/package prefix removal
            # and tolerate stray spaces.
            key = symbol_path.strip()
            if key in recs:
                return recs[key]
            available = ", ".join(sorted(recs.keys()))
            raise KeyError(
                f"Symbol not found: {symbol_path} in {file}. Available: {available}"
            )
        return recs[symbol_path]

    def get_source(self, record: SymbolRecord) -> str:
        """Return exact source code corresponding to the symbol."""
        mod = self._modules.get(record.file)
        if mod is None:
            raise KeyError(f"File not loaded: {record.file}")
        # Use wrapper to retrieve the node again by position.
        # For simplicity, slice the original text by lines; libcst nodes also
        # provide .get_code(), but we didn't keep the node here.
        src = self._sources[record.file]
        lines = src.splitlines()
        # Lines are 1-based in positions.
        start_line, start_col = record.position.start
        end_line, end_col = record.position.end
        if start_line == end_line:
            segment = lines[start_line - 1][start_col:end_col]
            return segment
        head = lines[start_line - 1][start_col:]
        middle = lines[start_line:end_line - 1]
        tail = lines[end_line - 1][:end_col]
        return "\n".join([head] + middle + [tail])

    def get_source_with_context(
        self,
        record: SymbolRecord,
        *,
        before: int = 2,
        after: int = 2,
    ) -> str:
        """Return source with a few lines of context around the symbol."""
        src = self._sources[record.file]
        lines = src.splitlines()
        start = max(record.position.start_line - 1 - before, 0)
        end = min(record.position.end_line - 1 + after, len(lines) - 1)
        return "\n".join(lines[start : end + 1])

    # --- private helpers ---
    def _abs(self, file_path: str | Path) -> Path:
        p = Path(file_path)
        if not p.is_absolute():
            p = (self.root / p).resolve()
        return p

    def _iter_py_files(self) -> List[Path]:
        files: List[Path] = []
        for dirpath, _dirnames, filenames in os.walk(self.root):
            for name in filenames:
                if name.endswith(".py"):
                    files.append(self._abs(Path(dirpath) / name))
        return files

    # --- modification API ---
    def transform_function(
            self,
            file_path: str | Path,
            symbol_path: str,
            transformer_cls,
            **kwargs
    ) -> tuple[SymbolRecord, str]:
        """
        Apply a CSTTransformer to the given symbol (function/method/class).
        Returns (new_record, new_code).
        """
        file = self._abs(file_path)
        if file not in self._modules:
            raise KeyError(f"File not loaded: {file}")

        module = self._modules[file]
        transformer = transformer_cls(symbol_path, **kwargs)
        new_module = module.visit(transformer)

        # Update cached source/module
        new_code = new_module.code
        self._sources[file] = new_code
        self._modules[file] = new_module

        # Rebuild index
        wrapper = cst_meta.MetadataWrapper(new_module)
        collector = _SymbolCollector(file)
        wrapper.visit(collector)
        self._index[file] = collector.records

        if symbol_path not in self._index[file]:
            raise KeyError(f"Symbol {symbol_path} not found after transform")
        return self._index[file][symbol_path], new_code

    def code_with_lineno(self, code: str, start_line: int = 1) -> str:
        """
        Print code with line numbers prefixed.
        """
        return "\n".join(
            f"{i:4d} | {line}" for i, line in enumerate(code.splitlines(), start=start_line)
        )

    def edit_symbol(
            self,
            file_path: str | Path,
            symbol_path: str,
            relative_line: int,
            new_code: str,
            mode: str = "insert",
    ) -> tuple[SymbolRecord, str]:
        """
        Edit a function/method/class body at a 'visible' relative line.
        The relative_line is counted from the line *after* the 'def' line (0-based),
        and includes lines inside nested blocks (if/for/while/with/try).
        """
        file = self._abs(file_path)
        if file not in self._modules:
            raise KeyError(f"File not loaded: {file}")
        module = self._modules[file]

        # Parse the incoming code into a statement list (supports multi-line statements)
        code_mod = cst.parse_module(new_code if new_code.endswith("\n") else new_code + "\n")
        stmts_to_insert = list(code_mod.body)

        class QuickEdit(cst.CSTTransformer):
            METADATA_DEPENDENCIES = (cst_meta.PositionProvider,)

            def __init__(self, target: str):
                self.target = target
                self._class_stack: list[str] = []

            def visit_ClassDef(self, node: cst.ClassDef):
                self._class_stack.append(node.name.value)

            def leave_ClassDef(self, original_node, updated_node):
                self._class_stack.pop()
                return updated_node

            def leave_FunctionDef(self, original_node: cst.FunctionDef, updated_node: cst.FunctionDef):
                # Only touch the target function/method
                qualname = ".".join(
                    self._class_stack + [original_node.name.value]) if self._class_stack else original_node.name.value
                if qualname != self.target:
                    return updated_node

                # Compute the target line using the original_node positions (metadata is bound to original)
                func_pos = self.get_metadata(cst_meta.PositionProvider, original_node)
                target_line = func_pos.start.line + 1 + relative_line  # first body line = def line + 1

                # Resolve on the original tree: map line -> (path, index in that body, did we hit a stmt start?)
                # Path is a list of statement indexes: starting from the function body, walk body.body[idx]
                # until reaching the statement that owns the target parent body.
                def resolve_insertion(original_body: cst.IndentedBlock, target_line: int):
                    path: list[int] = []  # statement index path

                    # Recursive descent: scan statements in order within the current body
                    def descend(body: cst.IndentedBlock) -> tuple[list[int], int, bool]:
                        stmts = list(body.body)
                        for i, stmt in enumerate(stmts):
                            pos = self.get_metadata(cst_meta.PositionProvider, stmt)
                            # If the target line is before or exactly at the start of this statement
                            if target_line <= pos.start.line:
                                # Hit: insert/replace at position i within the current body
                                return (path.copy(), i, target_line == pos.start.line)
                            # If the target line falls within this statement's range and it has a sub-body, descend
                            pos_end = self.get_metadata(cst_meta.PositionProvider, stmt).end
                            if pos.start.line < target_line <= pos_end.line:
                                # Only handle compound statements that own a body
                                if isinstance(stmt, (cst.If, cst.For, cst.While, cst.With, cst.Try)):
                                    path.append(i)
                                    return descend(stmt.body)
                                # Other compound types can be added if needed
                        # No earlier/deeper spot found: place at the end of the current body
                        return (path.copy(), len(stmts), False)

                    return descend(original_body)

                orig_path, insert_idx, hit_stmt_start = resolve_insertion(original_node.body, target_line)

                # Use the original path to locate the target parent body on updated_node
                def get_parent_body(updated_func: cst.FunctionDef, path: list[int]) -> tuple[
                    cst.CSTNode, cst.IndentedBlock]:
                    cur_owner: cst.CSTNode = updated_func  # current node that owns a body (function or stmt)
                    cur_body: cst.IndentedBlock = updated_func.body
                    for idx in path:
                        # Take the idx-th statement in the current body, then go into its .body
                        stmt = list(cur_body.body)[idx]
                        # These statements all have .body; resolve only descends into these kinds
                        cur_owner = stmt
                        cur_body = stmt.body  # type: ignore[attr-defined]
                    return cur_owner, cur_body

                owner_node, parent_body = get_parent_body(updated_node, orig_path)

                # Perform insert/replace on parent_body.body
                new_body_stmts = list(parent_body.body)
                if mode == "insert":
                    new_body_stmts[insert_idx:insert_idx] = stmts_to_insert
                elif mode == "replace":
                    if insert_idx >= len(new_body_stmts):
                        raise IndexError("relative_line points past the last statement; cannot replace.")
                    new_body_stmts[insert_idx:insert_idx + 1] = stmts_to_insert
                else:
                    raise ValueError("mode must be 'insert' or 'replace'.")

                new_parent_body = parent_body.with_changes(body=new_body_stmts)

                # Write the new body back to owner_node, then write owner_node back into the function
                if owner_node is updated_node:
                    return updated_node.with_changes(body=new_parent_body)
                else:
                    # owner_node is If/For/While/With/Try; all support .with_changes(body=...)
                    # We need to replace that statement back into its parent body, walking the path upward.
                    # For simplicity, rebuild from updated_node along the path and replace layer by layer.
                    def rebuild(updated_func: cst.FunctionDef, path: list[int],
                                new_leaf_body: cst.IndentedBlock) -> cst.FunctionDef:
                        if not path:
                            return updated_func.with_changes(body=new_leaf_body)
                        # Descend layer by layer, collecting body references along the way
                        bodies = [updated_func.body]
                        node = updated_func
                        cur_body = updated_func.body
                        for idx in path:
                            stmt = list(cur_body.body)[idx]
                            bodies.append(stmt.body)  # type: ignore[attr-defined]
                            cur_body = stmt.body  # type: ignore[attr-defined]
                        # Backtrack and replace
                        cur_replaced_body = new_leaf_body
                        for depth in reversed(range(len(path))):
                            idx = path[depth]
                            parent_body = bodies[depth]
                            stmts = list(parent_body.body)
                            stmt = stmts[idx]
                            stmt = stmt.with_changes(body=cur_replaced_body)  # type: ignore[attr-defined]
                            stmts[idx] = stmt
                            cur_replaced_body = parent_body.with_changes(body=stmts)
                        return updated_func.with_changes(body=cur_replaced_body)

                    return rebuild(updated_node, orig_path, new_parent_body)

        # Run the transform with MetadataWrapper (metadata is bound to the original tree)
        wrapper = cst_meta.MetadataWrapper(module)
        new_module = wrapper.visit(QuickEdit(symbol_path))
        new_src = new_module.code

        # Cache & rebuild index
        self._sources[file] = new_src
        self._modules[file] = new_module
        wrapper2 = cst_meta.MetadataWrapper(new_module)
        collector = _SymbolCollector(file)
        wrapper2.visit(collector)
        self._index[file] = collector.records

        return self._index[file][symbol_path], new_src

    def list_symbols(self, file_path: str | Path) -> list[SymbolRecord]:
        """
        Return all symbols (classes, functions, methods) in a given file.
        """
        file = self._abs(file_path)
        if file not in self._index:
            raise KeyError(f"File not indexed: {file}")
        return list(self._index[file].values())

    def get_all_files(self) -> list[Path]:
        """
        Return all indexed Python files in the project.
        """
        return list(self._index.keys())

# --- quick demo ---
if __name__ == "__main__":
    # import argparse

    # ap = argparse.ArgumentParser(description="Index a Python project with libcst.")
    # ap.add_argument("root", type=str, help="Project root folder")
    # ap.add_argument("file", type=str, help="Target file (relative to root or absolute)")
    # ap.add_argument("symbol", type=str, help="Symbol path like foo or ClassA.method_b")
    # ap.add_argument("--context", type=int, default=0, help="Context lines to include")
    # args = ap.parse_args()

    # print(args.root)
    parser = ProjectParser("./")
    parser.build_index()
    parser.edit_symbol(
        "editor.py",
        "Editor.get_lines_range",
        relative_line=0,
        new_code="print('Edited line inserted here')",
        mode="insert"
    )
    parser.edit_symbol(
        "editor.py",
        "Editor.get_lines_range",
        relative_line=4,
        new_code="print('Edited line inserted here')",
        mode="insert"
    )
    rec = parser.resolve("editor.py", "Editor.get_lines_range")
    print(rec)
    out = (
        parser.get_source(rec)
    )
    print(out)
    print("--- with context ---")
    print(parser.get_source_with_context(rec, before=3, after=3))
    print("--- Code with line numbers ---")
    print(rec.position.start_line)
    print(parser.code_with_lineno(out, start_line=rec.position.start_line))
    print(rec.position.end_line)
    print("-- List all symbols in editor.py --")
    print(parser.list_symbols("editor.py"))
    print("-- All indexed files --")
    print(parser.get_all_files())
