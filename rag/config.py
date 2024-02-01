from enum import Enum
from pydantic import BaseModel
from typing import List
from rag.common.types import ChunkingStrategy
from rag.common.constants import CHUNK_SIZE, WINDOW_SIZE


class ParseConfig(BaseModel):

    regex_footer: str = r"Text 086.*?(?=\n|$)"
    regex_leading_numerals: str = r"^\d+\w+"
    stop_words: List[str] = ["", "notes"]
    chunk_size: int = CHUNK_SIZE
    window_size: int = WINDOW_SIZE
    chunk_sizes: List[int] = [int(CHUNK_SIZE / 2**i) for i in range(3)]
    chunking_strategy: Enum = ChunkingStrategy.BASE
