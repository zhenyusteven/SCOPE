from abc import ABC, abstractmethod
from typing import List, Dict, Any, Optional, Tuple
from collections import namedtuple

ContextItem = namedtuple("ContextItem", ["role", "content"])

class RecapContext(ABC):
    def __init__(self):
        self.llm_context = []

    def inject_observation(self, obs: str) -> None:
        item = ContextItem(role="human", content="")
        self.llm_context.append(item)

    def inject_previous_plan(self, thought: str, subtasks: List[str]) -> None:
        item = ContextItem(role="human", content=f"Think: {thought}\nSubtasks: [{','.join(subtasks)}]")
        self.llm_context.append(item)

    def inject_init_plan(self, thought: str, subtasks: List[str]) -> None:
        item = ContextItem(role="", content="")
        self.llm_context.append(item)

    def call_llm_on_context(self):
        pass


class ReCAP(ABC):
    def __init__(self) -> None:
        self.context = RecapContext()
        self.actions = []
        self.observations = []

    def env_step(self, action: str) -> str:
        return "Obs"

    def is_primitive(self, action: str) -> bool:
        print(self.context.llm_context)
        print(f"Action: {action}")
        s = input("Input if primitive: ")
        return s == 'y'

    def init_plan(self) -> Tuple[str, List[str]]:
        return "Think", ["task1", "task2", "task3"]

    def refine_plan(self) -> Tuple[str, List[str]]:
        return "Think_Revised", ["task1", "task2", "task3"]

    def run(self):
        def _recap():
            thought, subtasks = self.init_plan()
            self.context.inject_init_plan(thought, subtasks)
            while subtasks:
                if self.is_primitive(subtasks[0]):
                    obs = self.env_step(subtasks[0])
                    self.actions.append(subtasks[0])
                    self.observations.append(obs)
                    self.context.inject_observation(obs)
                else:
                    _recap()
                    self.context.inject_previous_plan(thought, subtasks[1:])
                thought, subtasks = self.refine_plan()

        _recap()


