from abc import ABC, abstractmethod
from typing import List, Dict, Any, Optional, Tuple
from collections import namedtuple

from ..editor.editor import Editor


class RecapNode:
    def __init__(self, task_name: str, task_description: str = ""):
        self.children = []
        self.task_name = task_name
        self.task_description = task_description

    def add_child(self, child_node: 'RecapNode') -> None:
        self.children.append(child_node)


class RecapSWE:
    def __init__(self, task_name: str, task_description: str, fewshot_example: str) -> None:
        self.editor = Editor()
        self.task_name = task_name
        self.task_description = task_description
        self.fewshot_example = fewshot_example
        self.context_tree = RecapNode(task_name, task_description)
        self.llm_context = []

    def run(self) -> str:
        # Run ReCAP-SWE, with closure function inside
        # TODO: write main execution logic
        return self.editor.get_all_lines(with_line_id=False)

    def get_subtask_hierarchy(self) -> str:
        ans = []
        def dfs(node, depth):
            if node is None:
                return
            ans.append(' ' * (depth * 4) + self.task_name)
            for child in node.children:
                dfs(child, depth + 1)
        dfs(self.context_tree, 0)
        return '\n'.join(ans)

    def get_output_code(self, with_line_id: bool = False) -> str:
        return self.editor.get_all_lines(with_line_id=with_line_id)

