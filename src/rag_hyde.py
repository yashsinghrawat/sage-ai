"""
src/rag_hyde.py
─────────────────────────────────────────────────────────────────
Sage AI — HyDE RAG Pipeline (Hypothetical Document Embeddings)
 
The problem with Naive RAG:
  The user writes in casual, emotional, modern language ("I feel
  paralyzed and ashamed about my deadline"). Shlokas are written in
  formal, archaic, scriptural language. These two styles are
  semantically related but not always close in embedding space —
  this "vocabulary mismatch" caps retrieval quality.
 
The HyDE fix (Gao et al., 2022):
  1. Ask the LLM to imagine what an IDEAL Gita-style passage
     answering this problem would sound like — generate that
     hypothetical text.
  2. Embed THAT hypothetical passage instead of the raw user query.
  3. Retrieve shlokas similar to the hypothetical passage.
 
  Intuition: "scripture-style text" embeds closer to "scripture-style
  text" than "casual modern problem" does — so we bridge the gap by
  generating a fake scripture passage first, then searching for real
  ones like it.
 
Run:
    python src/rag_hyde.py
"""
 
import os
from pathlib import Path
 
import chromadb
from chromadb.config import Settings
from chromadb.utils import embedding_functions
from dotenv import load_dotenv
from groq import Groq
 
# Reuse the naive pipeline's helpers — same collection, same generation
from rag_naive import (
    build_context_block,
    generate_response,
    get_collection,
)
 
load_dotenv()
 
# ── Config ────────────────────────────────────────────────────────
GROQ_MODEL = os.getenv("GROQ_MODEL", "llama-3.1-8b-instant")
TOP_K = int(os.getenv("TOP_K_RETRIEVAL", "3"))
 
 
HYDE_PROMPT = """You are simulating a passage from the Bhagavad Gita. Given a \
person's problem, write a SHORT hypothetical verse (2-4 sentences) in the \
formal, philosophical style of the Gita — as if Krishna were directly \
addressing this exact situation.
 
Do not solve the problem casually. Write in the elevated, timeless register \
of scripture: address the listener as "you", speak of duty, the self, \
detachment, equanimity, action, or the eternal — whatever fits the emotional \
core of the problem.
 
Output ONLY the hypothetical verse text. No preamble, no explanation, no \
quotation marks."""
 
 
def generate_hypothetical_document(client: Groq, user_problem: str) -> str:
    """
    Step 1 of HyDE: ask the LLM to write a fake Gita-style passage
    that would be the 'ideal' answer to this problem.
    """
    response = client.chat.completions.create(
        model=GROQ_MODEL,
        messages=[
            {"role": "system", "content": HYDE_PROMPT},
            {"role": "user", "content": f"Problem: {user_problem}"},
        ],
        temperature=0.7,
        max_tokens=120,
    )
    return response.choices[0].message.content.strip()
 
 
def retrieve_with_hyde(
    collection: chromadb.Collection, hypothetical_doc: str, k: int = TOP_K
) -> list[dict]:
    """
    Step 2 of HyDE: embed the HYPOTHETICAL document (not the user's
    raw problem) and retrieve real shlokas similar to it.
    """
    results = collection.query(
        query_texts=[hypothetical_doc],
        n_results=k,
        include=["metadatas", "documents", "distances"],
    )
 
    retrieved = []
    for doc_id, meta, doc, distance in zip(
        results["ids"][0],
        results["metadatas"][0],
        results["documents"][0],
        results["distances"][0],
    ):
        relevance = 1 - (distance / 2)
        retrieved.append({
            "id": doc_id,
            "chapter": meta["chapter"],
            "verse": meta["verse"],
            "chapter_title": meta["chapter_title"],
            "sanskrit": meta["sanskrit"],
            "translation": meta["translation"],
            "emotions": meta["emotions"],
            "embed_text": doc,
            "relevance": round(relevance, 3),
        })
    return retrieved
 
 
def ask_sage_hyde(user_problem: str, k: int = TOP_K) -> dict:
    """
    Full HyDE RAG pipeline:
      user problem
        -> generate hypothetical Gita-style passage
        -> embed hypothetical passage, retrieve real shlokas
        -> generate final advice grounded in real shlokas
    """
    groq_api_key = os.getenv("GROQ_API_KEY")
    if not groq_api_key or groq_api_key == "your_groq_api_key_here":
        raise ValueError(
            "GROQ_API_KEY not set. Get a free key at https://console.groq.com "
            "and add it to your .env file."
        )
 
    groq_client = Groq(api_key=groq_api_key)
    collection = get_collection()
 
    # Step 1: generate hypothetical document
    hypothetical_doc = generate_hypothetical_document(groq_client, user_problem)
 
    # Step 2: retrieve using the hypothetical document
    shlokas = retrieve_with_hyde(collection, hypothetical_doc, k=k)
 
    # Step 3: generate final advice (same generation step as naive RAG)
    advice = generate_response(groq_client, user_problem, shlokas)
 
    return {
        "problem": user_problem,
        "hypothetical_doc": hypothetical_doc,
        "advice": advice,
        "shlokas_used": shlokas,
        "pipeline": "hyde_rag",
    }
 
 
def print_result(result: dict) -> None:
    print(f"\n💬 Problem: {result['problem']}")
    print(f"\n📝 Generated hypothetical passage (used for retrieval):")
    print(f"   \"{result['hypothetical_doc']}\"")
    print(f"\n🧘 Sage's advice:\n{result['advice']}")
    print(f"\n📖 Retrieved shlokas (relevance scores):")
    for s in result["shlokas_used"]:
        print(f"   [{s['id']}] Ch.{s['chapter']} V.{s['verse']} "
              f"— {s['relevance']:.3f} — themes: {s['emotions']}")
    print("─" * 60)
 
 
if __name__ == "__main__":
    print("🌿 Sage AI — HyDE RAG Pipeline\n")
 
    test_problems = [
        "I keep procrastinating on my final year project and the deadline is in 2 weeks. I feel paralyzed and ashamed.",
        "My best friend moved to another country and I feel really lonely and lost without them.",
    ]
 
    for problem in test_problems:
        result = ask_sage_hyde(problem)
        print_result(result)
 
    print("\n✅ Day 5 HyDE RAG pipeline complete.")