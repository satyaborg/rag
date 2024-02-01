from enum import Enum


class ChunkingStrategy(Enum):

    BASE = "base"
    WINDOW = "window"
    CHILD_TO_PARENT = "child_to_parent"


class EmbedProvider(Enum):

    OPENAI = "text-embedding-3-large"
    COHERE = "embed-english-v3.0"


class LLMModel(Enum):

    GPT4 = "gpt-4-turbo-preview"
    GPT35 = "gpt-3.5-turbo"
    MISTRAL = "mistral-medium"


class RetrievalStrategy(Enum):

    VECTOR = "vector"
    BM25 = "bm25"
    HYBRID = "hybrid"
