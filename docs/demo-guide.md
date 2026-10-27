# Interview demo guide (about 3 minutes)

A script for a screen-recorded or live walkthrough of the dashboard. Every number quoted below
was produced by the current models on the bundled sample data (in-memory BM25 backend); if you
retrain or ingest new documents, read the values off the screen instead.

## Before you start

```bash
python scripts/dev.py
```

Wait for the API to finish starting (about 25 s with saved indexes; about 50 s on a cold first
run), open http://localhost:5173, choose the dark theme (top-right toggle), and make sure the top
bar shows **API online**. Close other tabs; 1440 px wide or more reads best.

---

### 0:00–0:20 · Open the dashboard

**Show:** the Search page.

**Say:** "This is a hybrid document search engine. Each query goes through NER and intent
classification, then lexical BM25 and dense FAISS retrieval run in parallel, Reciprocal Rank
Fusion merges them, and an XGBoost learning-to-rank model produces the final order. The
dashboard shows every one of those stages."

### 0:20–0:50 · Run the query

**Type:** `my kubernetes pod keeps restarting` and press Enter.

**Point at:**
- **Intent:** *Troubleshooting*, ~98% confidence (XGBoost classifier; probability bars underneath).
- **Entity:** `kubernetes` · PRODUCT (spaCy plus a technology gazetteer).
- **Server latency:** tens of milliseconds (the first query after startup is slower).
- **Retrieval pipeline strip:** BM25 and FAISS sit in parallel with their own candidate counts;
  then RRF fusion, 22 features per candidate, and the reranker, each with its timing.

### 0:50–1:20 · Lexical vs semantic

**Click** the **BM25** tab.
**Say:** "BM25 is lexical: it scores word overlap. *Kubernetes Orchestration Basics* wins
because it repeats 'Kubernetes' most."

**Click** the **Semantic** tab.
**Say:** "FAISS compares sentence embeddings. *Fixing CrashLoopBackOff in Kubernetes* wins,
cosine about 0.79 against 0.49, because 'keeps restarting' means CrashLoopBackOff even
without shared words. **The two retrievers disagree on the #1 document.**"

### 1:20–1:50 · Fusion and reranking

**Click** the **Hybrid (RRF)** tab.
**Say:** "Reciprocal Rank Fusion only uses ranks, so incomparable score scales don't matter.
Here the two documents are #1 and #2 in opposite orders, so they get an *identical* RRF score
(1/61 + 1/62). Fusion alone cannot break the tie."

**Click** the **Final ranking** tab.
**Say:** "The XGBoost ranker scores each candidate from features. It keeps *Orchestration
Basics* first and CrashLoopBackOff second."

**Click** **Why this result?** on result #1, then on #2.
**Say:** "These are the exact features the model scored: BM25 score and rank, semantic
similarity, RRF score, term and entity overlap, category match, intent. CrashLoopBackOff has
far higher semantic similarity, yet ranks second. With only 34 training queries the model
leans on the hybrid score and rank, so it under-uses the semantic signal for troubleshooting
queries. That's an honest failure case, and because the features are exposed I can see why.
The fix would be more judged troubleshooting queries or an intent × semantic feature, not
hard-coding the order."

### 1:50–2:20 · Evaluation

**Click** **Analytics** in the sidebar.
**Point at** the Retrieval quality chart and table (NDCG@10): BM25 **0.803**, Semantic
**0.950**, Hybrid **0.928**, Hybrid + reranker **0.934**.

**Say:** "This is measured on 20 held-out, hand-labelled queries over a 76-document corpus, kept
separate from the reranker's training queries. On this small set, semantic retrieval alone
beats the hybrid and reranked pipeline. I report that as measured; with this little data it is
not a production benchmark, but it shows the whole pipeline is measurable."
Optionally point at **Reranker feature importance**: hybrid score and rank dominate, which
matches the behaviour we just saw.

### 2:20–2:45 · Data quality

**Click** **Clusters**, then the **Duplicates** tab.
**Say:** "During ingestion, DBSCAN runs over the embeddings. Two syndicated copies are
**confirmed duplicates**: cosine above 0.97 *and* token Jaccard above 0.9, so the later copy is
excluded from the index. *What Are Containers?* is only a **candidate**: semantically close
(0.91) but worded differently (Jaccard 0.35), so it stays."

**Click** the **Outliers** tab.
**Say:** "The sourdough, football and tomato articles are outliers, but so are some isolated
technical documents. Outliers are **flagged, not deleted**; they stay searchable."

### 2:45–3:00 · Close

**Say:** "So: FastAPI serves a pipeline of spaCy NER, an XGBoost intent classifier, BM25 on
Elasticsearch (with an in-memory fallback, shown on the System page), MiniLM embeddings in
FAISS, Reciprocal Rank Fusion and an XGBoost learning-to-rank reranker, with DBSCAN data-quality
checks at ingestion, offline evaluation, and this React dashboard that explains every result."

---

## If something goes wrong

| Symptom | Fix |
|---|---|
| Red "Unable to connect to the search API" | The API is not running or still starting; check the terminal running `scripts/dev.py`. |
| Analytics says "No evaluation report yet" | Click **Run evaluation** (about 10 s) or run `python scripts/evaluate.py`. |
| Different numbers than above | You ingested or retrained; the dashboard shows the live values, which are the ones to quote. |
| System page shows "In-memory fallback" | Expected without Elasticsearch. Start it with `docker compose up -d elasticsearch` and restart the API to use it. |

Backup query if time allows: `stop attackers from guessing user passwords`. BM25 ranks
*Preventing SQL Injection* first (shared words); semantic retrieval and the final ranking pick
*Defending Login Pages Against Credential Stuffing*.
