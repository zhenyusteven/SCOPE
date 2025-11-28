import os
from typing import Optional
from openai import OpenAI
from .llm import RecapLLM


class OpenAILLM(RecapLLM):
    """OpenAI LLM implementation using the OpenAI API."""
    
    def __init__(
        self,
        model_name: str = "gpt-4o",
        api_key: Optional[str] = None,
        temperature: float = 0.0,
        max_tokens: Optional[int] = None,
    ):
        self.model_name = model_name
        self.temperature = temperature
        self.max_tokens = max_tokens
        
        api_key = api_key or os.environ.get("OPENAI_API_KEY")
        if not api_key:
            raise ValueError("OPENAI_API_KEY must be provided or set as environment variable")
        
        self.client = OpenAI(api_key=api_key)
    
    def generate(self, prompt: str) -> str:
        messages=[
                {"role": "system", "content": "You are a coding patch generation assistant. You final goal is to generate a patch that can help solve the coding issue with the given problem statement and relevant context."},
                {"role": "user", "content": prompt}
            ]
        response = self.client.chat.completions.create(
            model=self.model_name,
            messages=[{"role": "user", "content": prompt}],
            temperature=self.temperature,
            max_tokens=self.max_tokens,
        )
        return response.choices[0].message.content

