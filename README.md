## Findings & Learnings

Sage AI is intentionally built as an experimentation-heavy RAG system. Beyond building the pipelines, the goal is to understand why retrieval succeeds or fails when grounding modern emotional problems in ancient philosophical text.

### 1. Naive RAG suffers from vocabulary mismatch

Modern emotional language (“I feel paralyzed and ashamed about my deadline”) embeds poorly against scripture-style text written in formal philosophical language.

**Observation:** With only 8 shlokas, retrieval scores were weak (~0.13–0.18 relevance) and semantically flat, often returning the “least bad” match rather than truly relevant guidance.

**Key takeaway:** Early bottlenecks came from retrieval quality, not LLM generation quality.

---

### 2. Corpus quality matters before retrieval sophistication

Expanding the curated dataset from **8 → 32 emotionally diverse shlokas** dramatically improved retrieval behavior.

**Observed improvement:**

* procrastination → duty, action, effort
* grief → impermanence, resilience
* loneliness → connection and presence

**Key takeaway:** Better corpus coverage improved retrieval more than prompt engineering.

---

### 3. HyDE significantly improved retrieval confidence

Using **Hypothetical Document Embeddings (HyDE)** improved semantic retrieval by first generating a Bhagavad Gita-style hypothetical answer and embedding that instead of the user's raw query.

**Example — procrastination query**

* **Naive RAG top relevance:** ~0.32
* **HyDE top relevance:** ~0.51

**Why this worked:** Ancient scripture and modern emotional language live in different semantic spaces. HyDE reduces this mismatch by transforming modern language into scripture-style language before retrieval.

---

### 4. HyDE revealed an unexpected failure mode: emotional drift

A loneliness query:

> “My best friend moved to another country and I feel really lonely and lost.”

produced a hypothetical passage focused on *departure*, *detachment*, and *the eternal*, which shifted retrieval toward:

* grief
* death
* impermanence

instead of:

* emotional connection
* companionship
* loneliness

This surfaced verses about mortality rather than friendship.

**Key takeaway:** Higher retrieval confidence does not necessarily imply better semantic alignment. Generated intermediary representations can subtly distort emotional framing.

**Planned fix:** Cross-encoder re-ranking using the **original user query** to recover semantically aligned verses from a broader retrieval pool.
