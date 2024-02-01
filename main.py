import argparse
import logging
from dotenv import load_dotenv

from rag.parser import DocParser, ParseConfig
from rag.embedding import Embedding
from rag.model import Model
from rag.indexer import Indexer
from rag.retriever import Retriever
from llama_index import ServiceContext
from rag.common.constants import DATASET_PATH, STORE_PATH
from rag.common.types import (
    LLMModel,
    EmbedProvider,
    ChunkingStrategy,
    RetrievalStrategy,
)

load_dotenv()
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)


def rag_pipeline(
    query_text: str,
    data_dir: str,
    store_dir: str,
    llm_model: LLMModel,
    embed_provider: EmbedProvider,
    chunking_strategy: ChunkingStrategy,
    retrieval_strategy: RetrievalStrategy,
):
    """Pipeline for RAG.

    Args:

        query_text (str): Query text.
        data_dir (str): Path to data directory.
        store_dir (str): Path to store directory.
        llm_model (LLMModel): Language model.
        embed_provider (EmbedProvider): Embedding provider.
        chunking_strategy (ChunkingStrategy): Chunking strategy.
        retrieval_strategy (RetrievalStrategy): Retrieval strategy.

    Returns:

            response (str): Response from RAG.
    """
    slug = f"{chunking_strategy.value}_{embed_provider.value}_{llm_model.value}_{retrieval_strategy.value}"

    # initialize embedding and generative model
    llm = Model(model_name=llm_model).model
    embed_model = Embedding(embed_provider=embed_provider).model
    service_context = ServiceContext.from_defaults(llm=llm, embed_model=embed_model)

    # document parser
    doc_parser = DocParser(config=ParseConfig())
    docs = doc_parser.load_docs(data_dir)
    nodes = doc_parser.get_nodes(docs=docs, chunking_strategy=chunking_strategy)

    # generate vector index from nodes
    indexer = Indexer(service_context)
    vector_index = indexer.get_vector_index(
        nodes=nodes, store_dir=f"{store_dir}/{slug}"
    )

    # create retriever and query engine
    retriever = Retriever(
        nodes=nodes,
        vector_index=vector_index,
        chunking_strategy=chunking_strategy,
        retrieval_strategy=retrieval_strategy,
        service_context=service_context,
    )
    query_engine = retriever.get_query_engine()
    response = query_engine.query(query_text)
    return response


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--query", type=str, required=True)
    parser.add_argument("--data_dir", default=DATASET_PATH, type=str, required=False)
    parser.add_argument("--store_dir", default=STORE_PATH, type=str, required=False)
    parser.add_argument(
        "--llm_model", default=LLMModel.MISTRAL, type=LLMModel, required=False
    )
    parser.add_argument(
        "--embed_provider",
        default=EmbedProvider.OPENAI,
        type=EmbedProvider,
        required=False,
    )
    parser.add_argument(
        "--chunking_strategy",
        default=ChunkingStrategy.BASE,
        type=ChunkingStrategy,
        required=False,
    )
    parser.add_argument(
        "--retrieval_strategy",
        default=RetrievalStrategy.HYBRID,
        type=RetrievalStrategy,
        required=False,
    )
    args = parser.parse_args()
    # retrieve and respond to query
    response = rag_pipeline(
        query_text=args.query,
        data_dir=args.data_dir,
        store_dir=args.store_dir,
        llm_model=args.llm_model,
        embed_provider=args.embed_provider,
        chunking_strategy=args.chunking_strategy,
        retrieval_strategy=args.retrieval_strategy,
    )
    logging.info(response)
