from abc import ABC, abstractmethod
from typing import Literal


class RecapLLM(ABC):
    @abstractmethod
    def generate(self, promopt: str) -> str:
        pass