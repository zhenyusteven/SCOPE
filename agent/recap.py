from abc import ABC, abstractmethod
from typing import List, Dict, Any, Optional, Tuple
from collections import namedtuple
from pathlib import Path

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from editor.editor import Editor
from editor.ast_parser import ProjectParser
from editor.code_tree import CodeSemanticTree, CodeNode


class RecapSWE(ABC):
    def __init__(self, task_name: str, task_description: str, fewshot_example: str, 
                 project_root: str | Path = None) -> None:
        self.editor = Editor()
        self.task_name = task_name
        self.task_description = task_description
        self.fewshot_example = fewshot_example
        self.most_recent_observation = None
        
        # Initialize code tree if project root is provided
        self.code_tree = None
        self.parser = None
        if project_root:
            self.initialize_code_tree(project_root)


    def initialize_code_tree(self, project_root: str | Path) -> None:
        """Initialize the code tree from the project root directory."""
        self.parser = ProjectParser(project_root)
        self.parser.build_index()
        self.code_tree = CodeSemanticTree(self.parser, project_name=Path(project_root).name)
        self.code_tree.build_from_parser(self.parser)

    @abstractmethod
    def is_primitive_node(self, node: CodeNode) -> bool:
        return node.is_leaf()

    def recursive_downward_prompt(self, node: CodeNode) -> str:
        prompt = f"Here is the coding problem description: {self.task_description}. 
        Now you are at {node.name}. It is a {node.kind} and the summary at this level is {node.summary}. 
        You can see these folders/files {node.children}. Please sort the children nodes from the most optimally relavant one to the least to accomplish the task."
        return prompt

    def nonleaf_backtracking_prompt(self, node: CodeNode, context: str, remaining_children: List[CodeNode]) -> str:
        prompt = f"You are at {node.name}. It is a {node.kind} and the summary at this level is {node.summary}. Previous context is {context}.
        Now you will return to the parent level. 
        Please sort the remaining children nodes {remaining_children} from the most optimally relavant one to the least to accomplish the task."
        
        return prompt

    def nonleaf_completion_prompt(self, node: CodeNode, context: str) -> str:
        prompt = f"You are at {node.name}. It is a {node.kind} and the summary at this level is {node.summary}. Previous context is {context}.
        Now you will return to the parent level.
        There is no remaining nodes. Check if the current context can help to achieve the task.
        If yes, return an "complete" string; otherwise, return a "incomplete" string."
        return prompt

    def leaf_backtracking_prompt(self, node: CodeNode, context: str, remaining_children: List[CodeNode]) -> str:
        prompt = f"You are at {node.name}. It is a {node.kind} and the summary at this level is {node.summary}. Previous context is {context}.
        Please determine if this node is helpful for completing tha task {self.task_name}. If it is, please return a updated context. 
        Then please sort the remaining children nodes {remaining_children} from the most optimally relavant one to the least to accomplish the task.
        If there are no remaining nodes, check if the current context can help to achieve the task.
        If yes, return an "complete" string; otherwise, return a "imcomplete" string.""
        return prompt

    def leaf_completion_prompt(self) -> str:
        prompt = f"You are at {node.name}. It is a {node.kind} and the summary at this level is {node.summary}. Previous context is {context}.
        Please determine if this node is helpful for completing tha task {self.task_name}. If it is, please return a updated context. 
        There is no remaining nodes. Check if the current context can help to achieve the task.
        If yes, return an "complete" string; otherwise, return a "incomplete" string."
        return prompt

    def call_llm(self, prompt: str) -> str:
        # TODO: setup llm call
        return "placeholder"

    def recap(self, node: CodeNode, context: str) -> str:
        prompt = self.recursive_downward_prompt()
        S = self.call_llm(prompt)
        # TODO: parse the context to get the sorted children node list

        while S != []:
            if self.is_primitive_node(S[0]):
                if len(S) > 1:
                    prompt = self.leaf_backtracking_prompt(node, context, S[1:]) 
                    context = self.call_llm(prompt)
                    # TODO: parse the context to get the updated context, the sorted children node list and the result of the backtracking

                else:
                    prompt = self.leaf_completion_prompt(node, context)
                    context = self.call_llm(prompt)
                    # TODO: parse the context to get the updated context and the result of the completion

            else:
                context = self.recap(S[0], context)

                if len(S) > 1:
                    prompt = self.nonleaf_backtracking_prompt(node, context, S[1:])
                    context = self.call_llm(prompt)
                    # TODO: parse the context to get the updated context, the sorted children node list
                else:
                    prompt = self.nonleaf_completion_prompt()
                    context = self.call_llm(prompt)
                    # TODO: parse the context to get the updated context and the result of the completion

        return context


    def run(self, context: str) -> str:
        # Run ReCAP-SWE, with a closure function inside
        
        return self.recap(self.code_tree.root, context)

        
    def get_subtask_hierarchy(self) -> str:
        ans = []
        def dfs(node: RecapNode, depth: int) -> None:
            if node is None:
                return
            ans.append(' ' * (depth * 4) + self.task_name)
            for child in node.children:
                dfs(child, depth + 1)
        dfs(self.context_tree, 0)
        return '\n'.join(ans)

    def get_output_code(self, with_line_id: bool = False) -> str:
        return self.editor.get_all_lines(with_line_id=with_line_id)

