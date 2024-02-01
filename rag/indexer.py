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

    def __init__(self, llm: str = None, embed_model: str = "") -> None:
        self._service_context = ServiceContext.from_defaults(
            llm=llm, embed_model=embed_model
        )

    @property
    def service_context(self) -> ServiceContext:
        """Get the ServiceContext object."""
        return self._service_context

    @service_context.setter
    def service_context(self, service_context: ServiceContext) -> None:
        """Set the ServiceContext object."""
        self._service_context = service_context

    def get_vector_index(
        self,
        nodes: Union[List[IndexNode], List[dict]] = None,
        store_dir: str = "./store",
    ) -> VectorStore:
        """Create vector index from nodes."""
        if os.path.exists(store_dir):
            print("Loading index from disk ..")
            storage_context = StorageContext.from_defaults(persist_dir=store_dir)
            index = load_index_from_storage(
                storage_context, service_context=self.service_context
            )
            return index
        else:
            print("Creating index ..")
            index = VectorStoreIndex(
                nodes, service_context=self.service_context, storage_context=None
            )
            index.storage_context.persist(persist_dir=store_dir)  # persist for later
            return index
