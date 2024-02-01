# general
import os
import logging
import time
import asyncio
import warnings
from dotenv import load_dotenv

warnings.filterwarnings("ignore", category=DeprecationWarning)

# rag specific
from rag.parser import DocParser, ParseConfig
from rag.embedding import Embedding
from rag.model import Model
from rag.indexer import Indexer
from rag.retriever import Retriever
from rag.evaluate import QADataset, Evaluator
from rag.common.constants import (
    CHUNKING_STRATEGIES,
    EMBED_PROVIDERS,
    LLM_MODELS,
    RETRIEVAL_STRATEGIES,
    EVAL_MODEL,
    DATASET_PATH,
    STORE_PATH,
    QA_DATASET_PATH,
)

# misc
import pandas as pd
from llama_index import ServiceContext

# NOTE: make sure .env has all API keys as per README.md
load_dotenv()
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)


async def eval():
    """Main entry function to run evaluation."""
    all_results = []
    start = time.time()
    logging.info(f"Starting evaluation ..")
    try:
        for chunking_strategy in CHUNKING_STRATEGIES:
            # 1. load, parse, and chunk input documents into nodes
            doc_parser = DocParser(config=ParseConfig())
            docs = doc_parser.load_docs(DATASET_PATH)

            logging.info(f"Applying chunking strategy: {chunking_strategy} ..")
            nodes = doc_parser.get_nodes(docs=docs, chunking_strategy=chunking_strategy)
            logging.info(f"Number of nodes: {len(nodes)}")

            # 1. generate or load the QA dataset
            if os.path.exists(QA_DATASET_PATH):
                logging.info("QA dataset exists ..")
                qa_dataset = QADataset.load_dataset(path=QA_DATASET_PATH)
            else:
                logging.info("QA dataset does not exist. Generating ..")
                os.makedirs("./qa_dataset", exist_ok=True)
                qa_dataset = QADataset.generate_dataset(
                    nodes=nodes,
                    path=QA_DATASET_PATH,
                )

            logging.info(f"Loaded QA dataset ..")
            for embed_provider in EMBED_PROVIDERS:
                embed_model = Embedding(embed_provider=embed_provider).model
                for llm_model in LLM_MODELS:
                    llm = Model(model_name=llm_model).model
                    logging.info(
                        f"Initialized LLM: {llm_model.value}, Embedding: {embed_provider.value} .."
                    )
                    for retrieval_strategy in RETRIEVAL_STRATEGIES:
                        slug = f"{chunking_strategy.value}_{embed_provider.value}_{llm_model.value}_{retrieval_strategy.value}"
                        logging.info(f"Starting evaluation: {slug}")

                        # 2. generate vector index from nodes
                        service_context = ServiceContext.from_defaults(
                            llm=llm, embed_model=embed_model
                        )
                        indexer = Indexer(service_context)
                        vector_index = indexer.get_vector_index(
                            nodes=nodes, store_dir=f"{STORE_PATH}/{slug}"
                        )
                        logging.info(f"Generated vector index ..")

                        # 3. create retriever and query engine
                        retriever = Retriever(
                            nodes=nodes,
                            vector_index=vector_index,
                            chunking_strategy=chunking_strategy,
                            retrieval_strategy=retrieval_strategy,
                            service_context=service_context,
                        )
                        retriever_chunk = retriever.get_retriever()
                        query_engine = retriever.get_query_engine()
                        logging.info(f"Initialized retriever ..")

                        # 4. evaluate RAG performance and compute metrics
                        llm_eval_model = Model(model_name=EVAL_MODEL).model
                        eval_service_context = ServiceContext.from_defaults(
                            llm=llm_eval_model
                        )
                        evaluator = Evaluator(service_context=eval_service_context)

                        # 4.a evaluate retrieval (metrics: MRR, Hit Rate)
                        retrieval_results = await evaluator.evaluate_retrieval(
                            qa_dataset=qa_dataset,
                            retriever=retriever_chunk,
                        )

                        # 4.b evaluate response (metrics: faithulness, relevancy)
                        response_results = await evaluator.evaluate_response(
                            qa_dataset=qa_dataset,
                            query_engine=query_engine,
                            max_queries=10,
                        )

                        # consolidate results
                        metrics = evaluator.get_eval_metrics(
                            name=slug,
                            chunking_strategy=chunking_strategy.value,
                            embed_provider=embed_provider.value,
                            llm_model=llm_model.value,
                            retrieval_strategy=retrieval_strategy.value,
                            retrieval_results=retrieval_results,
                            response_results=response_results,
                        )
                        all_results.append(metrics)
                        logging.info(f"Finished evaluation.")

    except Exception as e:
        logging.error(f"Error evaluating: {slug} - {e}")
        raise Exception(f"Error evaluating: {slug}") from e

    all_results_df = pd.concat(all_results)
    all_results_df.to_csv("./results/all_results.csv", index=False)  # export results
    logging.info(f"Total time taken: {(time.time() - start)/60:.2f} mins.")


if __name__ == "__main__":
    asyncio.run(eval())
