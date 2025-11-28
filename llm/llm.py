from abc import ABC, abstractmethod
from typing import Literal
from typing import Any
import json


class RecapLLM(ABC):
    """Abstract base class for LLM implementations."""
    
    @abstractmethod
    def generate(self, prompt: str) -> str:
        """Generate a response from the LLM given a prompt.
        """
        pass

def create_prompt(instance: dict[str, Any], context_data: dict[str, Any]) -> str:
    """Create a prompt for the LLM given an instance, tree, and context.
    """
    problem_statement = instance.get("problem_statement", "")
    prompt_parts = []
    prompt_parts.append("Generate a patch to fix the following issue:\n\n")
    prompt_parts.append(f"Problem Statement:\n{problem_statement}\n\n")
    prompt_parts.append(f"Relevant Context for generating the patch:\n{context_data['context']}\n\n")
    prompt_parts.append(f"Relevant Code Source for fixing the issue:\n{context_data['code_patch']}\n\n")
    prompt_parts.append("Generate a unified diff patch using the relevant context and code source to fix the issue. Only output the patch, no additional explanation.")
    return "".join(prompt_parts)


def create_llm(model_name: str, **kwargs) -> RecapLLM:
    """Create an LLM instance based on model name.
    """
    model_name_lower = model_name.lower()
    
    if model_name_lower in ('gpt-4o-mini', 'gpt-4', 'gpt-3.5', 'gpt-4o', 'gpt-4-turbo'):
        from .openai_llm import OpenAILLM
        return OpenAILLM(model_name=model_name, **kwargs)
    else:
        raise ValueError(
            f"Unknown model name: {model_name}. "
            f"Supported models: 'gpt-4o-mini'"
        )