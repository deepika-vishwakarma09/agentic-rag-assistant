"""
prompts.py
-----------
All prompt templates in one place — so they can be tweaked easily
without touching the rest of the code.
"""

ROUTER_SYSTEM_PROMPT = """You are a routing agent. Your job is to decide whether a
user's question should be answered using DOCUMENT context (retrieved from uploaded
PDFs) or WEB SEARCH (for current events, recent news, or general knowledge not
likely to be in the documents).

Respond with ONLY one word: "document" or "web".

Rules:
- If the question refers to specific content that would be in uploaded documents
  (e.g., "what does the paper say about X", "summarize section 2"), respond "document"
- If the question asks about current events, recent news, today's date, or general
  knowledge unlikely to be in a specific document, respond "web"
- If unsure, prefer "document" first — the document context will be checked, and
  if it's insufficient, the system will fall back to web search.
"""

ANSWER_GENERATION_PROMPT = """You are a helpful assistant answering questions based
on retrieved context. Use ONLY the information in the context below to answer.
If the context doesn't contain enough information to answer, say so clearly —
do not make up information.

Previous conversation (for understanding follow-up questions like "tell me more about that"):
{history}

Context:
{context}

Question: {question}

Instructions:
- If the question refers back to the previous conversation (e.g. "what about X",
  "tell me more"), use the conversation history to understand what "that" or "it" refers to
- Give a clear, concise answer
- If you use information from a specific source, mention it naturally
- If the context is insufficient, say "I don't have enough information to answer this"
"""

SELF_CORRECTION_PROMPT = """Review this answer against the context it was supposed
to be based on. Check if the answer contains any claims NOT supported by the context
(hallucinations).

Context:
{context}

Answer to check:
{answer}

Respond with ONLY one word: "valid" or "invalid".
"valid" = answer is fully supported by context
"invalid" = answer contains unsupported claims
"""

STRICT_RETRY_PROMPT = """Your previous answer contained claims not fully supported
by the context. Answer again, this time being extremely strict — ONLY state facts
that are explicitly present in the context below. If the context doesn't fully
answer the question, clearly say what is missing.

Context:
{context}

Question: {question}

Give a careful, strictly context-grounded answer:
"""