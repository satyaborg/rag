from typing import List
from rag.common.types import ChunkingStrategy as cs
from rag.common.types import RetrievalStrategy as rs
from rag.common.types import EmbedProvider, LLMModel


CHUNK_SIZE: int = 1024
WINDOW_SIZE: int = 3
# NOTE: Window and child-to-parent chunking strategies need
# to regen the QA dataset since the node IDs are different, which can be expensive
CHUNKING_STRATEGIES: List = [cs.BASE]  # cs.WINDOW, cs.CHILD_TO_PARENT,
EMBED_PROVIDERS: List = [EmbedProvider.OPENAI, EmbedProvider.COHERE]
LLM_MODELS: List = [LLMModel.MISTRAL, LLMModel.GPT35]
RETRIEVAL_STRATEGIES: List = [rs.VECTOR, rs.BM25, rs.HYBRID]
EVAL_MODEL: LLMModel = LLMModel.GPT4
DATASET_PATH: str = "./data/"
QA_DATASET_PATH: str = "./qa_dataset/motor_insurance_eval_dataset.json"
