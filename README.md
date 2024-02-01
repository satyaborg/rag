# ⛏️ Retrieval Augmented Generation (RAG) Pipeline & Evaluation

RAG is the application of information retrieval technique to generative models (such as LLMs) to produce highly relevant grounded reponses, conditioned on some external dataset or knowledge base.

Some key challenges with LLMs that are addressed by RAG:

- Hallucinations
- Adpatability, or lack of expert knowledge (e.g. internal company documentation)
- Limited information due to knowledge cut-off

However, it is not a silver bullet and requires careful consideration in terms of its components and architecture. For example, it is sensitive to things like:

- Quality of embeddings
- Base LLM model for response generation
- Chunking strategy
- "Lost in the Middle" issues where context/information from the middle is lost
- Retrieval strategies (vector-based, keyword-based, hybrid etc.) and more

## What is it?

`rag` is built on top of `llama-index` and provides a host of options to test different approaches for chunking, embedding, retrieval and generation. These are easily extensible to any future methods, models etc. due to modularity of the same.

- Chunking:
  - Base: Uses a basic sentence splitter for chunking along with additional sanitization and cleaning
  - Sentence window: Use a sliding window of sentences to chunk the document
  - Child-to-parent: Use variable length chunks and creates child-parent relationships, where retrieval is performed on the child nodes and generation is performed using all the context from parent nodes
- Embeddings:
  - OpenAI's `text-embedding-3-large`
  - Cohere's `embed-english-v3.0`
- LLMs:
  - OpenAI's `gpt-3.5-turbo` and `gpt-4-turbo-preview`
  - Mistral's `mistral-medium`
- Retrieval Strategies:
  - BM25 (Best matching algorithm)
  - Vector search
  - Hybrid (BM25 + Vector search)

A pipeline is a combination of any of the above.

## Getting Started

> Note: Tested on Ubuntu 20.04 with Python 3.11

1. Clone this repository and change directory into it
2. Install `poetry` and run `poetry install` to install all dependencies
3. Activate the environment using `poetry shell`
4. Run `python -m main` to execute the RAG pipelines and evaluate them

Make sure to add a `.env` file at the root directory containing valid `OPENAI_API_KEY`, `COHERE_API_KEY`, `MISTRAL_API_KEY`.

## QA Dataset

GPT-4 was used to generate a synthetic dataset comprised of two questions per node/chunk as the ground truth. The dataset can be accessed under `qa_dataset/`.

## Evaluation

> Note: Results will vary based on the data characteristics, dataset size, and other variables like chunk_size, similarity_top_k, and so on.

### Metrics

The following metrics were used to evaluate the pipelines for retrieval and response, respectively:

- Retrieval
  - Hit Rate: Fraction of samples in grouth truth that were retrieved
  - MRR: Mean Reciprocal Rank
- Response
  - Faithfulness: Percentage of samples where the generated response was the same as the ground truth
  - Relevancy: Percentage of samples where the generated response was relevant to the question

### Observations

1. When it comes to overall performance, including metrics for both retrieval and response, Mistral's `mistral-medium` with OpenAI's `text-embedding-3-large` using a hybrid approach (BM25 with vector search) performs the best. Using the same hybrid approach and embeddings but with GPT-3.5-turbo as the LLM, it comes as second best.

2. Mistral medium on the whole is more faithful as compared to GPT-3.5-turbo. However, only one combination i.e. `gpt-3.5-turbo` with Cohere's `embed-english-v3.0`, using hybrid retrieval, scored perfectly on both faithfulness and relevancy.

3. On retrieval strategies, the hybrid approach consistently outperforms the other two approaches.

4. Keyword-based traditional strategies unsurprisingly perform the worst. However, they tend to augment vector-based methods and improve performance overall as per (3).

Additionally, reranking methods can be employed to further boost performance.

Below is the overall evaluation results as sorted by hit rate, MRR, faithfulness and relevancy:

| LLM            | Embedding              | Chunking Strategy | Retrieval Strategy | Hit Rate |      MRR | Faithfulness | Relevancy |
| :------------- | :--------------------- | :---------------- | :----------------- | -------: | -------: | -----------: | --------: |
| mistral-medium | text-embedding-3-large | base              | hybrid             | 0.827922 | 0.635823 |            1 |       0.8 |
| gpt-3.5-turbo  | text-embedding-3-large | base              | hybrid             | 0.827922 | 0.635552 |          0.9 |       0.9 |
| gpt-3.5-turbo  | embed-english-v3.0     | base              | hybrid             | 0.798701 |    0.625 |            1 |         1 |
| mistral-medium | embed-english-v3.0     | base              | hybrid             | 0.798701 |    0.625 |            1 |       0.9 |
| mistral-medium | embed-english-v3.0     | base              | vector             | 0.668831 | 0.582792 |            1 |       0.9 |
| gpt-3.5-turbo  | embed-english-v3.0     | base              | vector             | 0.668831 | 0.582792 |            1 |       0.9 |
| gpt-3.5-turbo  | text-embedding-3-large | base              | vector             | 0.665584 | 0.590909 |          0.8 |       0.8 |
| mistral-medium | text-embedding-3-large | base              | vector             | 0.665584 | 0.589286 |            1 |       0.8 |
| mistral-medium | embed-english-v3.0     | base              | bm25               |  0.63961 | 0.574675 |            1 |       0.8 |
| mistral-medium | text-embedding-3-large | base              | bm25               |  0.63961 | 0.574675 |            1 |       0.7 |
| gpt-3.5-turbo  | text-embedding-3-large | base              | bm25               |  0.63961 | 0.574675 |          0.9 |       0.8 |
| gpt-3.5-turbo  | embed-english-v3.0     | base              | bm25               |  0.63961 | 0.574675 |          0.9 |       0.8 |

> Note: GPT-4-turbo was used as the LLM for response evaluation to measure faithfulness and relavancy. Only the first 10 samples were considered due to prohibitive cost while using the same as an evaluator. The `max_queries` argument can be overridden to evaluate more samples.

## References

- https://docs.llamaindex.ai/en/stable
- https://www.cs.otago.ac.nz/homepages/andrew/papers/2014-2.pdf
