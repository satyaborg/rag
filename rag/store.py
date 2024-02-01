import os
import qdrant_client
from llama_index.vector_stores.qdrant import QdrantVectorStore
from llama_index.storage.storage_context import StorageContext


class VectorStore:

    def __init__(self):
        self.client = qdrant_client.QdrantClient(location=":memory")

    def get_db(self, collection_name: str = None) -> StorageContext:
        """Get the QdrantVectorStore object."""
        vector_store = QdrantVectorStore(
            client=self.client, collection_name=collection_name
        )
        storage_context = StorageContext.from_defaults(vector_store=vector_store)
        return storage_context
