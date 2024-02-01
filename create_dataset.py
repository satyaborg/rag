from llama_index.evaluation import (
    generate_question_context_pairs,
    EmbeddingQAFinetuneDataset,
)
from llama_index.llms import OpenAI
from rag.common.types import ChunkingStrategy

from rag.parser import DocParser, ParseConfig

if __name__ == "__main__":
    llm = OpenAI(model="gpt-4")

    parse_config = ParseConfig()
    doc_parser = DocParser(config=parse_config)
    docs = doc_parser.load_docs("./data/")
    base_nodes = doc_parser.get_nodes(docs=docs, chunking_strategy=ChunkingStrategy.BASE)

    qa_dataset = generate_question_context_pairs(
        base_nodes, llm=llm, num_questions_per_chunk=2
    )
    qa_dataset.save_json("./qa_dataset/motor_insurance_eval_dataset.json")