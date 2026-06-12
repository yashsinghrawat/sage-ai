
"""
src/rag_naive.py
─────────────────────────────────────────────────────────────────
Sage AI — Naive RAG Pipeline
 
The simplest RAG approach:
1. Embed the user's query directly
2. Retrieve top-k most similar shlokas from ChromaDB
3. Stuff them into a prompt
4. Ask the LLM to generate advice grounded in those shlokas
 
This is the "naive" baseline — Day 7+ adds HyDE and re-ranking on
top of this same structure, so compare carefully once those exist.
 
Run:
    python src/rag_naive.py
"""
 
import os
from pathlib import Path
 
import chromadb
from chromadb.config import Settings
from chromadb.utils import embedding_functions
from dotenv import load_dotenv
from groq import Groq
 
load_dotenv()
 
# ── Config ────────────────────────────────────────────────────────
DATA_DIR = Path(__file__).parent.parent / "data"
CHROMA_PERSIST_DIR = str(DATA_DIR / "chroma_db")
COLLECTION_NAME = "gita_shlokas"
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "all-MiniLM-L6-v2")
GROQ_MODEL = os.getenv("GROQ_MODEL", "llama3-8b-8192")
TOP_K = int(os.getenv("TOP_K_RETRIEVAL", "3"))
 
 
SYSTEM_PROMPT = """You are Sage — a calm, wise guide who draws on the Bhagavad Gita \
to help people navigate modern problems. You are not preachy or religious; you are \
warm, direct, and practical.
 
You will be given:
1. A person's problem, described in their own words.
2. A set of relevant shlokas (verses) from the Bhagavad Gita, each with its \
   translation, chapter/verse reference, and Sanskrit text.
 
Your task:
- Acknowledge the person's situation with empathy, in 1-2 sentences.
- Choose the SINGLE most relevant shloka from the ones provided.
- Quote its translation and reference (Chapter X, Verse Y).
- Explain in plain, modern language what this verse means for THEIR specific \
  situation — connect it directly to what they described.
- End with one concrete, actionable takeaway.
 
Keep your entire response under 180 words. Do not use bullet points. Be a \
trusted friend who happens to know this text deeply, not a lecturer."""
 
 
def get_collection() -> chromadb.Collection:
    """Connects to the existing ChromaDB collection built by embeddings.py."""
    client = chromadb.PersistentClient(
        path=CHROMA_PERSIST_DIR,
        settings=Settings(anonymized_telemetry=False),  # silences telemetry warning
    )
    embedding_fn = embedding_functions.SentenceTransformerEmbeddingFunction(
        model_name=EMBEDDING_MODEL
    )
    return client.get_collection(name=COLLECTION_NAME, embedding_function=embedding_fn)
 
 
def retrieve(collection: chromadb.Collection, query: str, k: int = TOP_K) -> list[dict]:
    """
    Retrieves the top-k shlokas most similar to the query.
    Returns a list of dicts with id, metadata, and a 0-1 relevance score
    (cosine similarity — higher is better).
    """
    results = collection.query(
        query_texts=[query],
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
        # ChromaDB default space is squared L2 on normalized embeddings,
        # which is equivalent to: cosine_similarity = 1 - (distance / 2)
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
 
 
def build_context_block(shlokas: list[dict]) -> str:
    """Formats retrieved shlokas into a context block for the LLM prompt."""
    lines = []
    for s in shlokas:
        lines.append(
            f"— Chapter {s['chapter']}, Verse {s['verse']} "
            f"({s['chapter_title']}) [relevance: {s['relevance']}]\n"
            f"  Sanskrit: {s['sanskrit']}\n"
            f"  Translation: {s['translation']}\n"
            f"  Themes: {s['emotions']}"
        )
    return "\n\n".join(lines)
 
 
def generate_response(client: Groq, user_problem: str, shlokas: list[dict]) -> str:
    """Sends the problem + retrieved context to the LLM and returns advice."""
    context = build_context_block(shlokas)
 
    user_message = (
        f"My problem: {user_problem}\n\n"
        f"Relevant shlokas retrieved:\n\n{context}"
    )
 
    response = client.chat.completions.create(
        model=GROQ_MODEL,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_message},
        ],
        temperature=0.6,
        max_tokens=300,
    )
    return response.choices[0].message.content
 
 
def ask_sage(user_problem: str, k: int = TOP_K) -> dict:
    """
    Main entry point — the full naive RAG pipeline in one call.
    Returns a dict with the advice text and the shlokas used,
    so the API layer (Day 5+) can show citations in the UI.
    """
    groq_api_key = os.getenv("GROQ_API_KEY")
    if not groq_api_key or groq_api_key == "your_groq_api_key_here":
        raise ValueError(
            "GROQ_API_KEY not set. Get a free key at https://console.groq.com "
            "and add it to your .env file."
        )
 
    collection = get_collection()
    shlokas = retrieve(collection, user_problem, k=k)
 
    groq_client = Groq(api_key=groq_api_key)
    advice = generate_response(groq_client, user_problem, shlokas)
 
    return {
        "problem": user_problem,
        "advice": advice,
        "shlokas_used": shlokas,
        "pipeline": "naive_rag",
    }
 
 
def print_result(result: dict) -> None:
    print(f"\n💬 Problem: {result['problem']}")
    print(f"\n🧘 Sage's advice:\n{result['advice']}")
    print(f"\n📖 Retrieved shlokas (relevance scores):")
    for s in result["shlokas_used"]:
        print(f"   [{s['id']}] Ch.{s['chapter']} V.{s['verse']} "
              f"— {s['relevance']:.3f} — themes: {s['emotions']}")
    print("─" * 60)
 
 
if __name__ == "__main__":
    print("🌿 Sage AI — Naive RAG Pipeline\n")
 
    test_problems = [
        "I keep procrastinating on my final year project and the deadline is in 2 weeks. I feel paralyzed and ashamed.",
        "My best friend moved to another country and I feel really lonely and lost without them.",
    ]
 
    for problem in test_problems:
        result = ask_sage(problem)
        print_result(result)
 
    print("\n Day 3 naive RAG pipeline complete.")