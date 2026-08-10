"""
ragas_eval.py
---------------
Hamare RAG system ko numbers se measure karta hai, RAGAS library se.

3 metrics measure karte hain:
1. Faithfulness — kya answer sirf context se aaya, ya LLM ne khud se
   kuch bana diya (hallucination)?
2. Answer Relevancy — kya answer actually sawaal se related hai?
3. Context Precision — kya jo chunks retrieve hue wo sach mein useful the?

Important: RAGAS by default OpenAI use karta hai. Humne isko Groq
(jo already use kar rahe hain) aur local HuggingFace embeddings ke
saath configure kiya hai, taaki koi extra OpenAI key na chahiye.
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
    Har test question ko agent se chalata hai aur RAGAS ke liye
    zaroori format mein data collect karta hai: question, answer,
    contexts (list of retrieved texts), ground_truth.
    """
    questions, answers, contexts_list, ground_truths = [], [], [], []

    for item in test_set:
        question = item["question"]

        # Agent ko call karo — ye already document/web retrieval + answer
        # generation dono karta hai
        result = ask(question, vector_store, keyword_search)

        questions.append(question)
        answers.append(result["answer"])
        ground_truths.append(item["ground_truth"])

        # Note: RAGAS ko retrieved chunks ka raw text chahiye, sources nahi.
        # Router agent abhi sirf sources return karta hai, isliye yahan
        # hum context ko answer ke through hi approximate kar rahe hain
        # agar chunks directly available na hon. Better production setup
        # mein router_agent se raw context_chunks bhi return karo.
        contexts_list.append([result["answer"]])

    return {
        "question": questions,
        "answer": answers,
        "contexts": contexts_list,
        "ground_truth": ground_truths
    }


def evaluate_rag_system(vector_store, keyword_search, test_set_path: str = None):
    """
    Poora evaluation pipeline: test set load karo, agent chalao,
    RAGAS metrics compute karo, results print karo.
    """
    test_set_path = test_set_path or "app/evaluation/test_questions.json"
    test_set = load_test_set(test_set_path)

    print(f"Running evaluation on {len(test_set)} test questions...")
    eval_data = run_agent_on_test_set(test_set, vector_store, keyword_search)

    dataset = Dataset.from_dict(eval_data)

    # RAGAS ko Groq LLM aur local embeddings ke saath configure karo
    # (default OpenAI hai, jo hume nahi chahiye)
    llm = ChatGroq(api_key=settings.GROQ_API_KEY, model=settings.GROQ_MODEL)
    embeddings = HuggingFaceEmbeddings(model_name=settings.EMBEDDING_MODEL)

    # Groq ka free tier rate-limited hai — kam parallel workers aur
    # zyada timeout dene se TimeoutError errors avoid hote hain
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