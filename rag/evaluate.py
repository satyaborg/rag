import pandas as pd
from llama_index.llms import OpenAI
from llama_index.evaluation import (
    RelevancyEvaluator,
    FaithfulnessEvaluator,
    RetrieverEvaluator,
    BatchEvalRunner,
    generate_question_context_pairs,
    EmbeddingQAFinetuneDataset,
)
from typing import List, Union, Any
from llama_index.schema import IndexNode


class QADataset:

    @staticmethod
    def generate_dataset(
        nodes: Union[List[IndexNode], List[dict]],
        num_questions_per_chunk: int = 2,
        path: str = "./qa_dataset/data.json",
    ) -> dict:
        """Generate synthetic QA dataset."""
        qa_llm = OpenAI(model="gpt-4")  # use GPT-4 as the gold standard
        qa_dataset = generate_question_context_pairs(
            nodes, llm=qa_llm, num_questions_per_chunk=num_questions_per_chunk
        )
        qa_dataset.save_json(path)
        return qa_dataset

    @staticmethod
    def load_dataset(path: str) -> Any:
        qa_dataset = EmbeddingQAFinetuneDataset.from_json(path)
        return qa_dataset


class Evaluator:

    def __init__(self, service_context: Any) -> None:
        self.faithfulness_llm = FaithfulnessEvaluator(service_context=service_context)
        self.relevancy_llm = RelevancyEvaluator(service_context=service_context)

    @staticmethod
    def get_queries(qa_dataset: Any) -> List[str]:
        """Get queries from QA dataset."""
        queries = list(qa_dataset.queries.values())
        return queries

    async def evaluate_retrieval(self, qa_dataset, retriever):
        """Evaluate RAG retrieval."""
        retriever_evaluator = RetrieverEvaluator.from_metric_names(
            ["mrr", "hit_rate"], retriever=retriever
        )
        eval_results = await retriever_evaluator.aevaluate_dataset(qa_dataset)
        return eval_results

    async def evaluate_response(
        self,
        qa_dataset: Any,
        query_engine: Any,
        max_queries: int = 10,
        num_workers: int = 4,
    ):
        """Evaluate RAG response."""
        queries = self.get_queries(qa_dataset)
        batch_eval_queries = queries[:max_queries]
        runner = BatchEvalRunner(
            {"faithfulness": self.faithfulness_llm, "relevancy": self.relevancy_llm},
            workers=num_workers,
        )
        eval_results = await runner.aevaluate_queries(
            query_engine, queries=batch_eval_queries
        )
        return eval_results

    def get_eval_metrics(
        self,
        name: str,
        chunking_strategy: str,
        embed_provider: str,
        llm_model: str,
        retrieval_strategy: str,
        retrieval_results: List[dict],
        response_results: List[dict],
    ) -> pd.DataFrame:
        """Get retrieval metrics."""

        # get the retrieval evaluation performance
        metric_dicts = []
        for eval_result in retrieval_results:
            metric_dict = eval_result.metric_vals_dict
            metric_dicts.append(metric_dict)

        df = pd.DataFrame(metric_dicts)

        hit_rate = df["hit_rate"].mean()
        mrr = df["mrr"].mean()

        # get the response evaluation performance
        faithfulness_score = sum(
            result.passing for result in response_results["faithfulness"]
        ) / len(response_results["faithfulness"])
        relevancy_score = sum(
            result.passing for result in response_results["relevancy"]
        ) / len(response_results["relevancy"])

        # compile results into a single dataframe
        metric_df = pd.DataFrame(
            {
                "llm_model": [llm_model],
                "embed_provider": [embed_provider],
                "chunking_strategy": [chunking_strategy],
                "retrieval_strategy": [retrieval_strategy],
                "hit_rate": [hit_rate],
                "mrr": [mrr],
                "faithfulness_score": [faithfulness_score],
                "relevancy_score": [relevancy_score],
            }
        )
        # metric_df.to_json(
        #     f"./results/{name}_metrics.json", orient="records", lines=True
        # )
        return metric_df
