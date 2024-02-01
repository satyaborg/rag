import os
from typing import Any
from llama_index.embeddings.cohereai import CohereEmbedding
from llama_index.embeddings.openai import OpenAIEmbedding

from rag.common.types import EmbedProvider


class Embedding:

    def __init__(self, embed_provider: EmbedProvider = None) -> None:
        self.embed_provider = embed_provider
        self.model = self._load()

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(embed_provider={self.embed_provider.value})"

    def __call__(self, *args, **kwargs):
        return self.model(*args, **kwargs)

    def _load(self) -> Any:
        """Load Embeddings given a provider."""
        if self.embed_provider == EmbedProvider.COHERE:
            return CohereEmbedding(
                cohere_api_key=os.environ["COHERE_API_KEY"],
                model_name=self.embed_provider.value,
                input_type="search_query",
            )
        elif self.embed_provider == EmbedProvider.OPENAI:
            return OpenAIEmbedding(
                model=self.embed_provider.value,
                api_key=os.environ["OPENAI_API_KEY"],
            )
        else:
            raise ValueError(f"Invalid embedding provider: {self.embed_provider}")
