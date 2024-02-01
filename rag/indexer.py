import os
from typing import List, Union
from llama_index.schema import IndexNode
from llama_index import (
    VectorStoreIndex,
    ServiceContext,
    StorageContext,
    load_index_from_storage,
)
from llama_index.vector_stores.types import VectorStore


class Indexer:

    def __init__(self, service_context) -> None:
        self.service_context = service_context

    def get_vector_index(
        self,
        nodes: Union[List[IndexNode], List[dict]] = None,
        store_dir: str = "./store",
    ) -> VectorStore:
        """Create vector index from nodes."""
        if os.path.exists(store_dir):
            storage_context = StorageContext.from_defaults(persist_dir=store_dir)
            index = load_index_from_storage(
                storage_context, service_context=self.service_context
            )
            return index
        else:
            index = VectorStoreIndex(
                nodes, service_context=self.service_context, storage_context=None
            )
            index.storage_context.persist(persist_dir=store_dir)  # persist for later
            return index
