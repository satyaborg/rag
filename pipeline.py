import argparse
from rag.parser import DocParser, ParseConfig
from rag.embedding import Embedding
from rag.model import Model
from rag.indexer import Indexer
from rag.retriever import Retriever


def rag_pipeline(chunking_strategy, embed_provider, llm_model, retrieval_strategy):
    """"""
    embed_model = Embedding(embed_provider=embed_provider).model
    llm = Model(model_name=llm_model).model

    parse_config = ParseConfig()
    doc_parser = DocParser(config=parse_config)
    docs = doc_parser.load_docs("./data/")

    # chunk documents
    nodes = doc_parser.get_nodes(docs=docs, chunking_strategy=chunking_strategy)

    # create index
    indexer = Indexer(llm=llm, embed_model=embed_model)
    vector_index = indexer.get_vector_index(nodes=nodes, store_dir="./store/")
    # create retriever
    retriever = Retriever()
    retriever_chunk = retriever.get_retriever(
        nodes=nodes,
        vector_index=vector_index,
        chunking_strategy=chunking_strategy,
        retrieval_strategy=retrieval_strategy,
    )
    query_engine = retriever.get_query_engine(
        vector_index=vector_index,
        chunking_strategy=chunking_strategy,
        nodes=nodes,
        service_context=indexer.service_context,
    )
    query_engine, retriever_chunk


def run(query: str):
    """Run RAG pipeline."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--query", type=str, default="What is RedPajama-Data-v2?")
    parser.add_argument(
        "--chunking_strategy",
        type=str,
        default="paragraph",
        help="Chunking strategy to use.",
    )
    parser.add_argument(
        "--embed_provider",
        type=str,
        default="sentence-transformers",
        help="Embedding provider to use.",
    )
    parser.add_argument(
        "--llm_model", type=str, default="microsoft/DialoGPT-medium", help="LLM model."
    )
    parser.add_argument(
        "--retrieval_strategy",
        type=str,
        default="bm25",
        help="Retrieval strategy to use.",
    )
    args = parser.parse_args()

    query_engine, retriever = rag_pipeline(
        chunking_strategy=args.chunking_strategy,
        embed_provider=args.embed_provider,
        llm_model=args.llm_model,
        retrieval_strategy=args.retrieval_strategy,
    )
    response = query_engine.query(query)
    return response


# from llama_index.ingestion import IngestionPipeline

# pipeline = IngestionPipeline(
#     transformations=[TextCleaner(), text_splitter, embed_model],
#     vector_store=vector_store,
#     cache=ingest_cache,
# )


# from llama_index import SimpleDirectoryReader, ServiceContext, VectorStoreIndex
# from llama_index.embeddings import TogetherEmbedding
# from llama_index.llms import TogetherLLM


# # Provide a template following the LLM's original chat template.
# def completion_to_prompt(completion: str) -> str:
#   return f"<s>[INST] {completion} [/INST] </s>\n"


# def run_rag_completion(
#     document_dir: str,
#     query_text: str,
#     embedding_model: str ="togethercomputer/m2-bert-80M-8k-retrieval",
#     generative_model: str ="mistralai/Mixtral-8x7B-Instruct-v0.1"
#     ) -> str:
#     service_context = ServiceContext.from_defaults(
#         llm=TogetherLLM(
#             generative_model,
#             temperature=0.8,
#             max_tokens=256,
#             top_p=0.7,
#             top_k=50,
#             # stop=...,
#             # repetition_penalty=...,
#             is_chat_model=False,
#             completion_to_prompt=completion_to_prompt
#         ),
#         embed_model=TogetherEmbedding(embedding_model)
#     )
#     documents = SimpleDirectoryReader(document_dir).load_data()
#     index = VectorStoreIndex.from_documents(documents, service_context=service_context)
#     response = index.as_query_engine(similarity_top_k=5).query(query_text)

#     return str(response)


# query_text = "What is RedPajama-Data-v2? Describe in a simple sentence."
# document_dir = "./sample_doc_data"

# response = run_rag_completion(document_dir, query_text)
# print(response)
