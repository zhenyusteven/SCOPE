from abc import ABC, abstractmethod
from typing import List, Dict, Any, Optional, Tuple
from collections import namedtuple
from pathlib import Path

import json
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from editor.editor import Editor
from editor.ast_parser import ProjectParser
from editor.code_tree import CodeSemanticTree, CodeNode
from openai import OpenAI
from pydantic import BaseModel, ValidationError
from dotenv import load_dotenv

load_dotenv()
OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY")

class LLMResponse(BaseModel):
    context: str
    code_patch: str
    sorted_children: List[str]
    is_complete: bool = False

class RecapSWE(ABC):
    def __init__(self, task_name: str, task_description: str, fewshot_example: str, code_tree: CodeSemanticTree,
                 project_root: str | Path = None) -> None:
        self.code_tree = code_tree
        self.task_name = task_name
        self.task_description = task_description
        self.fewshot_example = fewshot_example

        self.llm = OpenAI(api_key=OPENAI_API_KEY)


    # def initialize_code_tree(self, project_root: str | Path) -> None:
    #     """Initialize the code tree from the project root directory."""
    #     self.parser = ProjectParser(project_root)
    #     self.parser.build_index()
    #     self.code_tree = CodeSemanticTree(self.parser, project_name=Path(project_root).name)
    #     self.code_tree.build_from_parser(self.parser)

    def is_primitive_node(self, node: CodeNode) -> bool:
        return node.is_leaf()

    def _format_children(self, child_ids: List[str]) -> str:
        details = []
        for cid in child_ids:
            child = self.code_tree.nodes.get(cid)
            if not child:
                continue
            details.append(f'{{"id": "{cid}", "name": "{child.name}", "kind": "{child.kind}"}}')
        return "[" + ", ".join(details) + "]"

    def _json_response_instruction(self, allowed_child_ids: List[str]) -> str:
        allowed = ", ".join(f'"{cid}"' for cid in allowed_child_ids) if allowed_child_ids else ""
        return (
            "Respond ONLY with valid JSON following this schema:\n"
            "{\n"
            '  "context": "<updated context string>",\n'
            '  "code_patch": "<actual code as a string>",\n'
            '  "sorted_children": ["<child_id>", ...],\n'
            '  "is_complete": <boolean: true or false>\n'
            "}\n"
            f'Only include ids from this list: [{allowed}].'
        )

    def _resolve_children(self, child_ids: List[str]) -> List[CodeNode]:
        resolved = []
        for cid in child_ids:
            node = self.code_tree.nodes.get(cid)
            if node:
                resolved.append(node)
        return resolved

    def recursive_downward_prompt(self, node: CodeNode, context: str, code_patch: str) -> str:
        children = node.children
        prompt = (f"Here is the coding problem description: {self.task_description}."
        f"Now you are at node named {node.name}, which is a {node.kind} and the summary at this level is {node.summary}."
        f"The code patch at this node is {node.source}."
        f"Previous context is {context} and relevant code patch is {code_patch}."
        f"Update the context by adding precise metadata that could help locate or modify the bug (file path, function/class name, approximate line span, why it matters, and the hypothesized change if any). Keep it concise but explicit."
        f"You can see these children nodes: {self._format_children(children)}. Please sort the children nodes from the most optimally relavant one to the least to accomplish the task."
        f"If the node summary is clearly unrelated to the task, stop exploring this branch by returning an empty sorted_children list."
        f"\n{self._json_response_instruction(children)}"
        f"Set is_complete to true only if you think the current context and code patch are sufficient to achieve the task; otherwise, set is_complete to false.")
        return prompt

    def nonleaf_backtracking_prompt(self, node: CodeNode, context: str, code_patch: str, remaining_children: List[CodeNode]) -> str:
        remaining_ids = [child.id for child in remaining_children]
        prompt = (f"You are back at {node.name}. It is a {node.kind} and the summary at this level is {node.summary}."
        f"The code patch at this node is {node.source}."
        f"Previous context is {context} and relevant code patch is {code_patch}."
        f"Ensure the context includes the most promising file/function/line locations and suspected edits gathered so far; merge or drop fluff to stay concise."
        f"Please sort the remaining sibling nodes {self._format_children(remaining_ids)} from the most optimally relavant one to the least to accomplish the task."
        f"\n{self._json_response_instruction(remaining_ids)}")
        return prompt

    def nonleaf_completion_prompt(self, node: CodeNode, context: str, code_patch: str) -> str:
        prompt = (f"You are at {node.name}. It is a {node.kind} and the summary at this level is {node.summary}."
        f"The code patch at this node is {node.source}."
        f"Previous context is {context} and relevant code patch is {code_patch}."
        f"Ensure the context highlights concrete locations (file paths, functions, line spans) and suggested modifications most relevant to the task."
        "Now you will return to the parent level. There is no remaining nodes. Check if the current context can help to achieve the task."
        "If yes, set is_complete to true; otherwise, set is_complete to false."
        f"\n{self._json_response_instruction([])}")
        return prompt

    def leaf_backtracking_prompt(self, leaf: CodeNode, parent: CodeNode, context: str, code_patch: str, remaining_siblings: List[CodeNode]) -> str:
        remaining_ids = [child.id for child in remaining_siblings]
        prompt = (f"You just visited leaf node {leaf.name}, which is a {leaf.kind} and the summary at this level is {leaf.summary}."
        f"The code patch at this node is {leaf.source}."
        f"Previous context is {context} and relevant code patch is {code_patch}."
        f"Determine whether this leaf contributes to solving task {self.task_name}; if it does, update the context and code patch accordingly, including file path, function/class, line span, and hypothesized fix."
        f"Determine whether this leaf contributes to solving task {self.task_name}; if it does, update the context and code patch accordingly."
        f"Now you are back at parent {parent.name}. Please sort the remaining sibling nodes {self._format_children(remaining_ids)} from the most optimally relavant one to the least to accomplish the task."
        f"If there are no remaining nodes, check if the current context can help to achieve the task."
        f"If yes, set is_complete to true; otherwise, set is_complete to false."
        f"\n{self._json_response_instruction(remaining_ids)}")
        return prompt

    def leaf_completion_prompt(self, leaf: CodeNode, parent: CodeNode, context: str, code_patch: str) -> str:
        prompt = (f"You are at leaf node {leaf.name}. It is a {leaf.kind} and the summary at this level is {leaf.summary}."
        f"The code patch at this node is {leaf.source}."
        f"Previous context is {context} and relevant code patch is {code_patch}."
        f"Please determine if this node is helpful for completing task {self.task_name}. If it is, update the context and code patch."
        f"If useful, add precise metadata: file path, function/class, approximate line span, and the concrete change you anticipate."
        f"There are no remaining sibling nodes under parent {parent.name}. Check if the current context can help to achieve the task."
        f"If yes, set is_complete to true; otherwise, set is_complete to false."
        f"\n{self._json_response_instruction([])}")
        return prompt

    def call_llm(self, prompt: str) -> str:
        response = self.llm.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": "You are a helpful programming assistant. You final goal is to identify the part of code in the project source code that can help solve the coding issue, and generate the comprehensive context for guiding a code agent to fix the code issue and the actual code patch from the project source code to fix the issue."},
                {"role": "user", "content": prompt + "\n\nPlease output the actual code patch from the project source code that could be modified to fix the issue."}
            ]
        )
        return response.choices[0].message.content.strip()
    
    def parse_llm_response(self, response: str) -> LLMResponse:
        try:
            raw = json.loads(response)
        except json.JSONDecodeError as exc:
            raise ValueError(f"LLM response is not valid JSON: {response}") from exc
        try:
            return LLMResponse(**raw)
        except ValidationError as exc:
            raise ValueError(f"LLM response does not match schema: {response}") from exc

    def _recap(self, node: CodeNode, context: str, code_patch: str) -> Tuple[str, str, bool]:
        prompt = self.recursive_downward_prompt(node, context, code_patch)
        response = self.call_llm(prompt)
        parsed = self.parse_llm_response(response)
        context = parsed.context
        code_patch = parsed.code_patch
        children = self._resolve_children(parsed.sorted_children)

        if parsed.is_complete or not children:
            return context, code_patch, parsed.is_complete

        while children:
            current = children[0]
            remaining = children[1:]

            if self.is_primitive_node(current):
                if remaining:
                    prompt = self.leaf_backtracking_prompt(current, node, context, code_patch, remaining)
                else:
                    prompt = self.leaf_completion_prompt(current, node, context, code_patch)
                response = self.call_llm(prompt)
                parsed = self.parse_llm_response(response)
                context = parsed.context
                code_patch = parsed.code_patch
                if parsed.is_complete:
                    return context, code_patch, True
                children = self._resolve_children(parsed.sorted_children)
                continue

            # non-leaf
            context, code_patch, done = self._recap(current, context, code_patch)
            if done:
                return context, code_patch, True

            remaining = children[1:]
            if remaining:
                prompt = self.nonleaf_backtracking_prompt(node, context, code_patch, remaining)
            else:
                prompt = self.nonleaf_completion_prompt(node, context, code_patch)
            response = self.call_llm(prompt)
            parsed = self.parse_llm_response(response)
            context = parsed.context
            code_patch = parsed.code_patch
            if parsed.is_complete:
                return context, code_patch, True
            children = self._resolve_children(parsed.sorted_children)

        return context, code_patch, False


    def recap(self, node: CodeNode, context: Optional[str] = None, code_patch: Optional[str] = None) -> Tuple[str, str]:
        """Depth-first traversal that builds concise context and code patch guidance."""
        context = context or ""
        code_patch = code_patch or ""
        context, code_patch, _ = self._recap(node, context, code_patch)
        return context, code_patch


    def run(self, context: str = None, code_patch: str = None) -> Tuple[str, str]:
        # Run ReCAP-SWE, with a closure function inside

        return self.recap(self.code_tree.root, context, code_patch)

        
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
