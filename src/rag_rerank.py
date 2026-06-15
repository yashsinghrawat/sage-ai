"""
src/rag_rerank.py
─────────────────────────────────────────────────────────────────
Sage AI — Cross-Encoder Re-Ranking RAG Pipeline

The problem HyDE can introduce:
  HyDE retrieves based on similarity to a GENERATED hypothetical
  passage, not the user's actual words. If that passage drifts in
  tone (e.g. "departure" -> grief-over-death cluster, when the user
  meant "moved abroad"), retrieval drifts with it. We saw exactly
  this in Day 5 — Ch.2 V.20 (death) outranked Ch.6 V.30 (connection)
  for a loneliness query.

The re-ranking fix:
  1. Retrieve a WIDE net of candidates (top-10) using the FAST
     bi-encoder (same embedding model as naive RAG) — cast a wide
     net so the correct verse is likely SOMEWHERE in the candidates.
  2. Re-score each candidate using a CROSS-ENCODER: a model that
     takes (query, document) as a PAIR and directly predicts how
     relevant they are to EACH OTHER. This is slower per-pair but
     much more accurate, because it reads the actual user query —
     no hypothetical-passage drift.
  3. Re-sort by cross-encoder score, take the new top-k.

Bi-encoder vs cross-encoder:
  - Bi-encoder: embeds query and docs SEPARATELY, compares vectors.
    Fast (one embedding per doc, reusable), less accurate.
  - Cross-encoder: embeds (query, doc) TOGETHER, one forward pass
    per pair. Slow (must re-run for every candidate), more accurate.
  This is why we use bi-encoder for the wide retrieval (step 1) and
  cross-encoder only on the small candidate set (step 2) — best of
  both worlds.

Run:
    python src/rag_rerank.py
"""

import os

from chromadb.utils import embedding_functions
from dotenv import load_dotenv
from groq import Groq
from sentence_transformers import CrossEncoder

from rag_naive import ( generate_response, get_collection, )

load_dotenv()

# ── Config ────────────────────────────────────────────────────────
GROQ_MODEL = os.getenv("GROQ_MODEL", "llama-3.1-8b-instant")
TOP_K_FINAL = int(os.getenv("TOP_K_RERANK", "3"))
TOP_K_CANDIDATES = 10  # wide net for step 1, before re-ranking

# ms-marco-MiniLM-L-6-v2: small, fast, well-regarded cross-encoder
# trained specifically for query-passage relevance ranking
CROSS_ENCODER_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"

_cross_encoder = None  # lazy-loaded singleton, model load is slow


def get_cross_encoder() -> CrossEncoder:
    """Loads the cross-encoder model once and caches it."""
    global _cross_encoder
    if _cross_encoder is None:
        _cross_encoder = CrossEncoder(CROSS_ENCODER_MODEL)
    return _cross_encoder


def retrieve_candidates(collection, query: str, k: int = TOP_K_CANDIDATES) -> list[dict]:
    """
    Step 1: wide retrieval using the fast bi-encoder (same as naive RAG).
    Cast a wide net — the cross-encoder will narrow it down.
    """
    results = collection.query(
        query_texts=[query],
        n_results=k,
        include=["metadatas", "documents", "distances"],
    )

    candidates = []
    for doc_id, meta, doc, distance in zip(
        results["ids"][0],
        results["metadatas"][0],
        results["documents"][0],
        results["distances"][0],
    ):
        bi_encoder_score = 1 - (distance / 2)
        candidates.append({
            "id": doc_id,
            "chapter": meta["chapter"],
            "verse": meta["verse"],
            "chapter_title": meta["chapter_title"],
            "sanskrit": meta["sanskrit"],
            "translation": meta["translation"],
            "emotions": meta["emotions"],
            "embed_text": doc,
            "bi_encoder_score": round(bi_encoder_score, 3),
        })
    return candidates


def rerank_candidates(query: str, candidates: list[dict], k: int = TOP_K_FINAL) -> list[dict]:
    """
    Step 2: re-score each candidate with the cross-encoder, reading
    the candidate's embed_text directly against the ORIGINAL query.
    Step 3: re-sort and return the new top-k.
    """
    cross_encoder = get_cross_encoder()

    # Cross-encoder expects pairs of (query, document) strings
    pairs = [(query, c["embed_text"]) for c in candidates]
    cross_scores = cross_encoder.predict(pairs)

    for candidate, score in zip(candidates, cross_scores):
        candidate["normalized_score"] = round(float(score), 3)

        # Normalize for readability (0–1-ish)
        candidate["normalized_score"] = round(
            1 / (1 + abs(float(score))),
            3
        )
        # 'relevance' is the field rag_naive.build_context_block expects
        candidate["relevance"] = candidate["cross_encoder_score"]

    reranked = sorted(candidates, key=lambda c: c["cross_encoder_score"], reverse=True)
    return reranked[:k]


def ask_sage_rerank(user_problem: str, k: int = TOP_K_FINAL) -> dict:
    """
    Full re-ranking RAG pipeline:
      user problem
        -> retrieve top-10 candidates (bi-encoder, fast)
        -> re-score all 10 against the query (cross-encoder, accurate)
        -> take new top-k
        -> generate final advice grounded in re-ranked shlokas
    """
    groq_api_key = os.getenv("GROQ_API_KEY")
    if not groq_api_key or groq_api_key == "your_groq_api_key_here":
        raise ValueError(
            "GROQ_API_KEY not set. Get a free key at https://console.groq.com "
            "and add it to your .env file."
        )

    groq_client = Groq(api_key=groq_api_key)
    collection = get_collection()

    # Step 1: wide retrieval
    candidates = retrieve_candidates(collection, user_problem, k=TOP_K_CANDIDATES)

    # Steps 2-3: cross-encoder re-ranking
    shlokas = rerank_candidates(user_problem, candidates, k=k)

    # Final generation — same as naive RAG, grounded in re-ranked shlokas
    advice = generate_response(groq_client, user_problem, shlokas)

    return {
        "problem": user_problem,
        "advice": advice,
        "shlokas_used": shlokas,
        "all_candidates": candidates,  # kept for inspection/debugging
        "pipeline": "rerank_rag",
    }


def print_result(result: dict) -> None:
    print(f"\n💬 Problem: {result['problem']}")
    print(f"\n🧘 Sage's advice:\n{result['advice']}")
    print(f"\n📖 Final shlokas after re-ranking (cross-encoder scores):")
    for s in result["shlokas_used"]:
        print(f"   [{s['id']}] Ch.{s['chapter']} V.{s['verse']} "
              f"— cross: {s['cross_encoder_score']:.3f} "
              f"(bi-encoder was: {s['bi_encoder_score']:.3f}) "
              f"— themes: {s['emotions']}")

    # Show how much re-ranking REORDERED things vs the original bi-encoder order
    print(f"\n   Original bi-encoder top-{TOP_K_FINAL} (for comparison):")
    bi_sorted = sorted(result["all_candidates"], key=lambda c: c["bi_encoder_score"], reverse=True)
    for s in bi_sorted[:TOP_K_FINAL]:
        print(f"   [{s['id']}] Ch.{s['chapter']} V.{s['verse']} "
              f"— bi: {s['bi_encoder_score']:.3f}")
    print("─" * 60)


if __name__ == "__main__":
    print("🌿 Sage AI — Cross-Encoder Re-Ranking RAG Pipeline\n")
    print("(First run downloads the cross-encoder model — ~70MB, one-time)\n")

    test_problems = [
        "I keep procrastinating on my final year project and the deadline is in 2 weeks. I feel paralyzed and ashamed.",
        "My best friend moved to another country and I feel really lonely and lost without them.",
    ]

    for problem in test_problems:
        result = ask_sage_rerank(problem)
        print_result(result)

    print("\Day 6 re-ranking RAG pipeline complete.")