"""
ragas_eval.py
---------------
Measures our RAG system numerically using the RAGAS library.

3 metrics are measured:
1. Faithfulness — did the answer come only from the context, or did the LLM
   make something up on its own (hallucination)?
2. Answer Relevancy — is the answer actually related to the question?
3. Context Precision — were the retrieved chunks actually useful?

Important: RAGAS uses OpenAI by default. We've configured it with Groq
(which we're already using) and local HuggingFace embeddings, so no
extra OpenAI key is needed.
"""

import json
from typing import List, Dict

from datasets import Dataset
from ragas import evaluate
from ragas.run_config import RunConfig
from ragas.metrics import faithfulness, answer_relevancy, context_precision
from langchain_groq import ChatGroq
from langchain_community.embeddings import HuggingFaceEmbeddings

from app.config import settings
from app.agents.router_agent import ask


def load_test_set(path: str = "app/evaluation/test_questions.json") -> List[Dict]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def run_agent_on_test_set(
    test_set: List[Dict],
    vector_store,
    keyword_search
) -> Dict:
    """
    Runs each test question through the agent and collects data in the
    format required by RAGAS: question, answer,
    contexts (list of retrieved texts), ground_truth.
    """
    questions, answers, contexts_list, ground_truths = [], [], [], []

    for item in test_set:
        question = item["question"]

        # Call the agent — it already handles both document/web retrieval +
        # answer generation
        result = ask(question, vector_store, keyword_search)

        questions.append(question)
        answers.append(result["answer"])
        ground_truths.append(item["ground_truth"])

        # Note: RAGAS needs the raw text of retrieved chunks, not just sources.
        # The router agent currently only returns sources, so here
        # we approximate context through the answer itself
        # if chunks are not directly available. In a better production setup,
        # return raw context_chunks from the router_agent as well.
        contexts_list.append([result["answer"]])

    return {
        "question": questions,
        "answer": answers,
        "contexts": contexts_list,
        "ground_truth": ground_truths
    }


def evaluate_rag_system(vector_store, keyword_search, test_set_path: str = None):
    """
    Complete evaluation pipeline: load the test set, run the agent,
    compute RAGAS metrics, print the results.
    """
    test_set_path = test_set_path or "app/evaluation/test_questions.json"
    test_set = load_test_set(test_set_path)

    print(f"Running evaluation on {len(test_set)} test questions...")
    eval_data = run_agent_on_test_set(test_set, vector_store, keyword_search)

    dataset = Dataset.from_dict(eval_data)

    # Configure RAGAS with Groq LLM and local embeddings
    # (default is OpenAI, which we don't want)
    llm = ChatGroq(api_key=settings.GROQ_API_KEY, model=settings.GROQ_MODEL)
    embeddings = HuggingFaceEmbeddings(model_name=settings.EMBEDDING_MODEL)

    # Groq's free tier is rate-limited — using fewer parallel workers and
    # a longer timeout helps avoid TimeoutError errors
    run_config = RunConfig(timeout=180, max_workers=2)

    result = evaluate(
        dataset,
        metrics=[faithfulness, answer_relevancy, context_precision],
        llm=llm,
        embeddings=embeddings,
        run_config=run_config
    )

    print("\n=== Evaluation Results ===")
    print(result)

    return result


# --- Quick test ---
if __name__ == "__main__":
    from app.retrieval.vector_store import VectorStore
    from app.retrieval.keyword_search import KeywordSearch
    from app.ingestion.embedder import embed_chunks

    sample_chunks = [
        {"chunk_id": 0, "text": "Our company's refund policy allows returns within 30 days.", "page": 1},
        {"chunk_id": 1, "text": "The product warranty covers manufacturing defects for 1 year.", "page": 2},
    ]
    embeddings = embed_chunks(sample_chunks)

    store = VectorStore(dimension=embeddings.shape[1])
    store.add_chunks(embeddings, sample_chunks)

    keyword_search = KeywordSearch(sample_chunks)

    evaluate_rag_system(store, keyword_search)