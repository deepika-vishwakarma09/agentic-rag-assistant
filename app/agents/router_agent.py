"""
router_agent.py
------------------
Ye project ka "agentic" core hai.

Flow:
1. User ka question aata hai
2. Router LLM decide karta hai: "document" se jawab doon ya "web" search karoon
3. Uske hisaab se retrieval hoti hai (FAISS ya Tavily)
4. Retrieved context ke saath LLM final answer generate karta hai
5. Answer ke saath source citation bhi milta hai

LangGraph isliye use kiya kyunki ye multi-step agent flows ko
"graph" ki tarah define karne deta hai — har step ek node hai,
aur edges decide karte hain agla step kya hoga (conditional routing).
"""

from typing import TypedDict, List, Dict, Literal
from langgraph.graph import StateGraph, END

from app.retrieval.vector_store import VectorStore
from app.retrieval.keyword_search import KeywordSearch, merge_search_results
from app.retrieval.reranker import rerank
from app.ingestion.embedder import embed_query
from app.agents.web_search_tool import web_search
from app.agents.self_correction import verify_answer, retry_with_strict_grounding
from app.generation.llm_client import call_llm
from app.generation.prompts import ROUTER_SYSTEM_PROMPT, ANSWER_GENERATION_PROMPT


# --- Agent state: ye har node ke beech mein pass hoti hai ---
class AgentState(TypedDict):
    question: str
    history: str             # formatted previous conversation text
    route: str              # "document" ya "web"
    context_chunks: List[Dict]
    answer: str
    sources: List[str]
    was_corrected: bool     # true agar self-correction ne answer ko dobara banaya


# --- Node 1: Router — decide karta hai document ya web ---
def route_question(state: AgentState) -> AgentState:
    decision = call_llm(
        system_prompt=ROUTER_SYSTEM_PROMPT,
        user_prompt=state["question"],
        temperature=0.0  # routing decision consistent honi chahiye
    ).lower().strip()

    # Safety: agar LLM kuch aur bol de, default "document" rakho
    route = "web" if "web" in decision else "document"

    state["route"] = route
    return state


# --- Node 2a: Document retrieval (hybrid: semantic + keyword, then reranked) ---
def retrieve_from_documents(
    state: AgentState,
    vector_store: VectorStore,
    keyword_search: KeywordSearch
) -> AgentState:
    question = state["question"]

    # Stage 1a: Semantic search (FAISS) — meaning-based
    query_embedding = embed_query(question)
    semantic_results = vector_store.search(query_embedding, top_k=10)

    # Stage 1b: Keyword search (BM25) — exact-word-based
    keyword_results = keyword_search.search(question, top_k=10)

    # Stage 1c: Merge both (hybrid search) — best of both worlds
    hybrid_results = merge_search_results(
        semantic_results, keyword_results, semantic_weight=0.6, top_k=10
    )

    # Stage 2: Rerank the hybrid results with a cross-encoder for final precision
    final_results = rerank(question, hybrid_results, top_k=3)

    state["context_chunks"] = final_results
    state["sources"] = [f"Page {r['page']}" for r in final_results]
    return state


# --- Node 2b: Web search ---
def retrieve_from_web(state: AgentState) -> AgentState:
    results = web_search(state["question"], max_results=3)

    state["context_chunks"] = results
    state["sources"] = [r["source"] for r in results]
    return state


# --- Node 3: Answer generation ---
def generate_answer(state: AgentState) -> AgentState:
    # Sare chunks ka text jodo ek context string mein
    context_text = "\n\n".join([c["text"] for c in state["context_chunks"]])

    if not context_text.strip():
        state["answer"] = "Mujhe is sawaal ka jawab dene ke liye kaafi context nahi mila."
        return state

    prompt = ANSWER_GENERATION_PROMPT.format(
        context=context_text,
        question=state["question"],
        history=state.get("history", "No previous conversation.")
    )

    answer = call_llm(
        system_prompt="You are a helpful, accurate assistant.",
        user_prompt=prompt,
        temperature=0.3
    )

    state["answer"] = answer
    return state


# --- Node 4: Self-correction — verify answer, retry once if hallucination detected ---
def verify_and_correct(state: AgentState) -> AgentState:
    context_text = "\n\n".join([c["text"] for c in state["context_chunks"]])

    if not context_text.strip():
        # Context hi nahi tha to verify karne ka koi matlab nahi
        state["was_corrected"] = False
        return state

    is_valid = verify_answer(context_text, state["answer"])

    if not is_valid:
        corrected_answer = retry_with_strict_grounding(context_text, state["question"])
        state["answer"] = corrected_answer
        state["was_corrected"] = True
    else:
        state["was_corrected"] = False

    return state


# --- Conditional edge: route decide karke sahi retrieval node pe bhejo ---
def decide_retrieval_path(state: AgentState) -> Literal["document", "web"]:
    return state["route"]


def build_agent_graph(vector_store: VectorStore, keyword_search: KeywordSearch):
    """
    Poora LangGraph graph banata hai aur compile karke return karta hai.
    vector_store aur keyword_search dono yahan pass karte hain kyunki
    dono already-loaded documents rakhte hain (hybrid search ke liye zaroori).
    """
    graph = StateGraph(AgentState)

    graph.add_node("router", route_question)
    graph.add_node("doc_retrieval", lambda s: retrieve_from_documents(s, vector_store, keyword_search))
    graph.add_node("web_retrieval", retrieve_from_web)
    graph.add_node("generate", generate_answer)
    graph.add_node("verify", verify_and_correct)

    graph.set_entry_point("router")

    # Router ke baad, route ke hisaab se doc ya web retrieval
    graph.add_conditional_edges(
        "router",
        decide_retrieval_path,
        {
            "document": "doc_retrieval",
            "web": "web_retrieval"
        }
    )

    graph.add_edge("doc_retrieval", "generate")
    graph.add_edge("web_retrieval", "generate")
    graph.add_edge("generate", "verify")
    graph.add_edge("verify", END)

    return graph.compile()


def ask(
    question: str,
    vector_store: VectorStore,
    keyword_search: KeywordSearch,
    conversation_store=None,
    session_id: str = "default"
) -> Dict:
    """
    Main entry point — question do, poora answer + sources + route wapas milega.

    conversation_store diya ho to previous history bhi use hogi (follow-up
    questions samajhne ke liye), aur is exchange ko history mein save bhi
    kar diya jaayega.
    """
    agent = build_agent_graph(vector_store, keyword_search)

    history_text = "No previous conversation."
    if conversation_store is not None:
        history_text = conversation_store.get_history_as_text(session_id)

    result = agent.invoke({
        "question": question,
        "history": history_text,
        "route": "",
        "context_chunks": [],
        "answer": "",
        "sources": [],
        "was_corrected": False
    })

    if conversation_store is not None:
        conversation_store.add_message(session_id, "user", question)
        conversation_store.add_message(session_id, "assistant", result["answer"])

    return {
        "answer": result["answer"],
        "sources": result["sources"],
        "route_used": result["route"],
        "was_corrected": result["was_corrected"]
    }


# --- Quick test ---
if __name__ == "__main__":
    from app.ingestion.embedder import embed_chunks

    # Dummy document data setup karo testing ke liye
    sample_chunks = [
        {"chunk_id": 0, "text": "Our company's refund policy allows returns within 30 days.", "page": 1},
        {"chunk_id": 1, "text": "The product warranty covers manufacturing defects for 1 year.", "page": 2},
    ]
    embeddings = embed_chunks(sample_chunks)

    store = VectorStore(dimension=embeddings.shape[1])
    store.add_chunks(embeddings, sample_chunks)

    keyword_search = KeywordSearch(sample_chunks)

    from app.memory.conversation_store import ConversationStore
    conv_store = ConversationStore()
    session_id = "test_session"

    # Test 1: Document-based question
    print("=== Test 1: Document question ===")
    result1 = ask("What is the refund policy?", store, keyword_search, conv_store, session_id)
    print(f"Route used: {result1['route_used']}")
    print(f"Answer: {result1['answer']}")
    print(f"Sources: {result1['sources']}")
    print(f"Self-corrected: {result1['was_corrected']}")

    # Test 2: Follow-up question (isko history samajhni chahiye ki "it" = warranty ya refund)
    print("\n=== Test 2: Follow-up question (memory test) ===")
    result2 = ask("What about the warranty period for it?", store, keyword_search, conv_store, session_id)
    print(f"Route used: {result2['route_used']}")
    print(f"Answer: {result2['answer']}")
    print(f"Sources: {result2['sources']}")
    print(f"Self-corrected: {result2['was_corrected']}")

    # Test 3: Web-based question
    print("\n=== Test 3: Web question ===")
    result3 = ask("What is the latest news in AI today?", store, keyword_search, conv_store, session_id)
    print(f"Route used: {result3['route_used']}")
    print(f"Answer: {result3['answer']}")
    print(f"Sources: {result3['sources']}")
    print(f"Self-corrected: {result3['was_corrected']}")