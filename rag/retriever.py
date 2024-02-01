from typing import List, Union, Any
from llama_index.schema import IndexNode
from llama_index.vector_stores.types import VectorStore
from llama_index import ServiceContext
from llama_index.query_engine import RetrieverQueryEngine
from llama_index.postprocessor import MetadataReplacementPostProcessor
from llama_index.retrievers import BaseRetriever, RecursiveRetriever, BM25Retriever

from rag.common.types import ChunkingStrategy, RetrievalStrategy


class HybridRetriever(BaseRetriever):

    def __init__(self, vector_retriever: Any, bm25_retriever: Any) -> None:
        self.vector_retriever = vector_retriever
        self.bm25_retriever = bm25_retriever
        super().__init__()

    def _retrieve(self, query: str, **kwargs) -> Union[List[IndexNode], List[dict]]:
        bm25_nodes = self.bm25_retriever.retrieve(query, **kwargs)
        vector_nodes = self.vector_retriever.retrieve(query, **kwargs)

        # NOTE: combine the two lists of nodes
        all_nodes = []
        node_ids = set()
        for n in bm25_nodes + vector_nodes:
            if n.node.node_id not in node_ids:
                all_nodes.append(n)
                node_ids.add(n.node.node_id)
        return all_nodes


class Retriever:

    def __init__(
        self,
        nodes: Union[List[IndexNode], List[dict]],
        vector_index: VectorStore,
        chunking_strategy: ChunkingStrategy,
        retrieval_strategy: RetrievalStrategy,
        service_context: ServiceContext,
    ) -> None:
        self.top_k = 2
        self.response_mode = "compact"
        self.nodes = nodes
        self.vector_index = vector_index
        self.chunking_strategy = chunking_strategy
        self.retrieval_strategy = retrieval_strategy
        self.service_context = service_context

    def get_base_retriever(self) -> Any:
        """Retriever for the base approach."""
        base_retriever = self.vector_index.as_retriever(similarity_top_k=self.top_k)
        return base_retriever

    def get_recursive_retriever(self) -> Any:
        """Retriever for the recursive (child-to-parent) approach."""
        vector_retriever = self.vector_index.as_retriever(similarity_top_k=self.top_k)
        if isinstance(self.nodes, dict):
            retriever_chunk = RecursiveRetriever(
                "vector",
                retriever_dict={"vector": vector_retriever},
                node_dict=self.nodes,
                verbose=True,
            )
        else:
            raise ValueError(f"Invalid node type: {type(self.nodes)}")
        return retriever_chunk

    def get_sentence_window_retriever(self) -> Any:
        """Retriever for the sentence window approach."""
        sentence_window_retriever = self.vector_index.as_retriever(
            similarity_top_k=self.top_k,
            node_postprocessors=[
                MetadataReplacementPostProcessor(target_metadata_key="window")
            ],
        )
        return sentence_window_retriever

    def get_child_to_parent_query_engine(self) -> Any:
        """Query engine for the child-to-parent chunking approach."""
        retriever_chunk = self.get_recursive_retriever()
        query_engine = RetrieverQueryEngine.from_args(
            retriever_chunk,
            service_context=self.service_context,
            verbose=True,
            response_mode=self.response_mode,
        )
        return query_engine

    def get_sentence_window_query_engine(self) -> Any:
        """Query engine for the sentence window approach."""
        query_engine = self.vector_index.as_query_engine(
            similarity_top_k=self.top_k,
            node_postprocessors=[
                MetadataReplacementPostProcessor(target_metadata_key="window")
            ],
        )
        return query_engine

    def get_base_query_engine(self) -> Any:
        """Query engine for the base approach."""
        query_engine = self.vector_index.as_query_engine(similarity_top_k=self.top_k)
        return query_engine

    def get_query_engine(self) -> Any:
        """Get query engine based on chunking strategy."""
        if self.chunking_strategy == ChunkingStrategy.BASE:
            return self.get_base_query_engine()

        elif self.chunking_strategy == ChunkingStrategy.WINDOW:
            return self.get_sentence_window_query_engine()

        elif self.chunking_strategy == ChunkingStrategy.CHILD_TO_PARENT:
            return self.get_child_to_parent_query_engine()

        else:
            raise ValueError(f"Invalid chunking strategy: {self.chunking_strategy}")

    def get_retriever(self) -> Any:
        """Get retriever based on retrieval strategy."""
        if self.retrieval_strategy == RetrievalStrategy.VECTOR:
            if self.chunking_strategy == ChunkingStrategy.BASE:
                return self.get_base_retriever()

            elif self.chunking_strategy == ChunkingStrategy.WINDOW:
                return self.get_sentence_window_retriever()

            elif self.chunking_strategy == ChunkingStrategy.CHILD_TO_PARENT:
                return self.get_recursive_retriever()

            else:
                raise ValueError(f"Invalid chunking strategy: {self.chunking_strategy}")

        elif self.retrieval_strategy == RetrievalStrategy.BM25:
            return BM25Retriever.from_defaults(
                nodes=self.nodes, similarity_top_k=self.top_k
            )

        elif self.retrieval_strategy == RetrievalStrategy.HYBRID:
            return HybridRetriever(
                vector_retriever=self.get_base_retriever(),
                bm25_retriever=BM25Retriever.from_defaults(
                    nodes=self.nodes, similarity_top_k=self.top_k
                ),
            )

        else:
            raise ValueError(f"Invalid retrieval strategy: {self.retrieval_strategy}")
