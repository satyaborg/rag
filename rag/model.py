import os
from typing import Any
from llama_index.llms import OpenAI, MistralAI
from rag.common.types import LLMModel


class Model:

    def __init__(self, model_name: LLMModel, temperature: float = 0.0) -> None:
        self.model_name = model_name
        self.temperature = temperature
        self.model = self._load()

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(name={self.model_name.value})"

    def __call__(self, *args: Any, **kwargs: Any) -> Any:
        return self.model(*args, **kwargs)

    def _load(self) -> Any:
        """Loads LLM."""
        if self.model_name == LLMModel.MISTRAL:
            return MistralAI(
                model=self.model_name.value,
                temperature=self.temperature,
                api_key=os.environ["MISTRAL_API_KEY"],
            )
        elif self.model_name in [LLMModel.GPT35, LLMModel.GPT4]:
            return OpenAI(
                model=self.model_name.value,
                temperature=self.temperature,
                api_key=os.environ["OPENAI_API_KEY"],
            )
        else:
            raise ValueError(f"Invalid LLM model name: {self.model_name}")
