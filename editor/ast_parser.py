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
        self.root = Path(root)
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

        # 更新缓存
        new_code = new_module.code
        self._sources[file] = new_code
        self._modules[file] = new_module

        # 重新生成 index
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

        # 解析插入代码为语句列表（支持多行语句）
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
                # 仅处理目标函数/方法
                qualname = ".".join(
                    self._class_stack + [original_node.name.value]) if self._class_stack else original_node.name.value
                if qualname != self.target:
                    return updated_node

                # 计算目标行（以 original_node 的行号为准，metadata 对 original 有效）
                func_pos = self.get_metadata(cst_meta.PositionProvider, original_node)
                target_line = func_pos.start.line + 1 + relative_line  # def 后第一行 = +1

                # 在 original 树上解析“行号 -> (路径, 在该 body 的索引, 是不是精确命中一条语句起始行)”
                # 路径由一串语句下标组成，含义是：从函数体开始，依次取 body.body[idx]，直到到达目标父 body 的拥有者语句
                def resolve_insertion(original_body: cst.IndentedBlock, target_line: int):
                    path: list[int] = []  # 语句下标路径

                    # 递归下降：在当前 body 中按语句顺序扫描
                    def descend(body: cst.IndentedBlock) -> tuple[list[int], int, bool]:
                        stmts = list(body.body)
                        for i, stmt in enumerate(stmts):
                            pos = self.get_metadata(cst_meta.PositionProvider, stmt)
                            # 如果目标行在此语句之前或正好命中该语句起始行
                            if target_line <= pos.start.line:
                                # 命中：在当前 body 的 i 位置插入/替换
                                return (path.copy(), i, target_line == pos.start.line)
                            # 若目标行在该语句范围内，并且该语句有子 body，则进入子 body
                            pos_end = self.get_metadata(cst_meta.PositionProvider, stmt).end
                            if pos.start.line < target_line <= pos_end.line:
                                # 仅处理带 body 的复合语句
                                if isinstance(stmt, (cst.If, cst.For, cst.While, cst.With, cst.Try)):
                                    path.append(i)
                                    return descend(stmt.body)
                                # 其它复合语句类型可按需扩展
                        # 没有更早/更内层的位置：落在当前 body 末尾
                        return (path.copy(), len(stmts), False)

                    return descend(original_body)

                orig_path, insert_idx, hit_stmt_start = resolve_insertion(original_node.body, target_line)

                # 根据 original 的路径，在 updated_node 上定位目标父 body
                def get_parent_body(updated_func: cst.FunctionDef, path: list[int]) -> tuple[
                    cst.CSTNode, cst.IndentedBlock]:
                    cur_owner: cst.CSTNode = updated_func  # 当前“拥有 body”的节点（函数或某个语句）
                    cur_body: cst.IndentedBlock = updated_func.body
                    for idx in path:
                        # 取当前 body 里的第 idx 个语句，再进入它的 .body
                        stmt = list(cur_body.body)[idx]
                        # 这些语句都拥有 .body（对应 resolve 时我们只会在这些类型上下钻）
                        cur_owner = stmt
                        cur_body = stmt.body  # type: ignore[attr-defined]
                    return cur_owner, cur_body

                owner_node, parent_body = get_parent_body(updated_node, orig_path)

                # 在 parent_body.body 上做插入/替换
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

                # 把新的 body 写回 owner_node，再把 owner_node 写回函数
                if owner_node is updated_node:
                    return updated_node.with_changes(body=new_parent_body)
                else:
                    # owner_node 是 If/For/While/With/Try，这些都有 .with_changes(body=...)
                    # 需要把该语句替换回其父 body。沿 path 回溯一层一层替换。
                    # 为了简单，我们从 updated_node 再按 path 重建，逐层替换。
                    def rebuild(updated_func: cst.FunctionDef, path: list[int],
                                new_leaf_body: cst.IndentedBlock) -> cst.FunctionDef:
                        if not path:
                            return updated_func.with_changes(body=new_leaf_body)
                        # 逐层下钻，收集沿途 body 引用
                        bodies = [updated_func.body]
                        node = updated_func
                        cur_body = updated_func.body
                        for idx in path:
                            stmt = list(cur_body.body)[idx]
                            bodies.append(stmt.body)  # type: ignore[attr-defined]
                            cur_body = stmt.body  # type: ignore[attr-defined]
                        # 回溯替换
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

        # 用 MetadataWrapper 驱动变换（metadata 绑定在 original 树上）
        wrapper = cst_meta.MetadataWrapper(module)
        new_module = wrapper.visit(QuickEdit(symbol_path))
        new_src = new_module.code

        # 缓存 & 重建索引
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