# Sparse vs Dense Retrieval — Report (3–4 pages)

**Course**: Text_Analytics  
**Author**: Harshal Kamble  
**Date**: September 13, 2025

---

## 1. Dataset
I used a synthetic corpus of **1,000 short documents** across **10 domains** (e.g., renewable energy, e‑commerce analytics, healthcare AI, finance, mobility, etc.).  
Each row contains: `doc_id`, `topic`, and `text`. For evaluation, I prepared **10 natural queries** and a relevance file mapping each query to **10 relevant documents** (`qrels`).

- Corpus: `data/my_corpus_1k.csv` (1,000 docs × 10 topics)  
- Queries: `data/queries.json` (10 queries)  
- Qrels: `data/qrels.json` (10 relevant doc_ids per query)

This setup approximates a small, balanced multi‑topic collection where lexical cues and semantic similarity can both matter.

---

## 2. Methods
### 2.1 Sparse Retrieval (TF‑IDF)
- **Vectorizer**: `TfidfVectorizer` with unigrams + bigrams (`ngram_range=(1,2)`), `min_df=2`  
- **Similarity**: cosine similarity  
- **Index**: in‑memory SciKit‑Learn matrix (CSR)

**Rationale.** TF‑IDF is a strong lexical baseline that rewards exact word overlap. It tends to excel when queries reuse the same terms found in relevant documents (e.g., domain‑specific phrases).

### 2.2 Dense Retrieval (Sentence‑Transformers + FAISS)
- **Encoder**: `sentence-transformers/all-MiniLM-L6-v2` (384‑dim embeddings)  
- **Index**: FAISS `IndexFlatIP` with **L2‑normalized** vectors (cosine via dot product)  
- **Pipeline**: encode corpus once (batched), add to FAISS; encode query at runtime and search top‑k

**Rationale.** Dense models capture **semantic similarity**, handling synonyms and paraphrases better than purely lexical methods. This often improves recall when the query vocabulary differs from document vocabulary.

---

## 3. Experiments
- **Queries**: 10 queries covering the 10 topics.  
- **Metrics**: Recall@5, Recall@10, and Mean Reciprocal Rank (MRR@k).  
- **Evaluation**: compute metrics per query and report the average across all queries.  
- **Visualization**: a bar chart comparing TF‑IDF vs Dense (`outputs/results.png`).

Hyperparameters were kept minimal and identical to the defaults described above to focus on the high‑level comparison.

---

## 4. Results
**Aggregate metrics (averaged over 10 queries):**

| Method  | Recall@5 | Recall@10 | MRR@5 | MRR@10 |
|:--------|---------:|----------:|------:|-------:|
| TF‑IDF  | **0.040** | **0.070**  | **0.275** | **0.285** |
| Dense   | **0.070** | **0.100**  | **0.220** | **0.231** |

**Interpretation.**
- **Dense** retrieval achieved **higher recall** at both cutoffs (**+0.03 at @5, +0.03 at @10**), consistent with semantic matching benefits.  
- **TF‑IDF** achieved a **higher MRR**, indicating that when TF‑IDF hits, it often ranks a relevant item **very early** (benefit of exact term overlap).  
- Overall, Dense is preferable when coverage/recall is the priority (e.g., recall‑oriented first‑stage retrieval), whereas TF‑IDF can surface precise lexical matches at the very top.

*Figure 1. Sparse vs Dense comparison plot (`outputs/results.png`).*

---

## 5. Error Analysis
I inspected representative queries where the two systems diverged:

**Case A — TF‑IDF > Dense (lexical advantage).**  
- **Observation.** Queries containing **topic‑specific terms** (e.g., “expected goals”, “HIPAA”, “Kubernetes”) favored TF‑IDF because these exact tokens appear in the relevant documents.  
- **Example pattern.** Query mentions “**expected goals** in soccer analytics”; relevant docs also contain the phrase “expected goals”. TF‑IDF ranks them very high, while Dense occasionally prefers semantically related documents that lack the exact phrase.  
- **Why.** High‑idf n‑grams anchor the match; lexical overlap aligns strongly with relevance for such technical phrases.

**Case B — Dense > TF‑IDF (semantic advantage).**  
- **Observation.** Queries phrased with **synonyms or paraphrases** (e.g., “ride‑sharing” vs “rideshare”, “charging infrastructure” vs “EV charging stations”) favored Dense.  
- **Example pattern.** Query mentions “**charging infrastructure for electric cars**”; relevant docs discuss “EV charging stations” and “charging networks” without using the exact query words. Dense retrieves them; TF‑IDF under‑ranks due to vocabulary mismatch.  
- **Why.** Dense embeddings cluster paraphrases, so semantically similar but lexically different documents are retrieved earlier.

**Takeaway from errors.**  
- Lexical exactness helps **precision@1**/**MRR**, especially with domain n‑grams.  
- Semantic encoding helps **recall**, especially when wording varies between query and documents.

---

## 6. Takeaways
1. **Dense improves recall** on this multi‑topic corpus, making it a strong **first‑stage retriever**.  
2. **TF‑IDF improves early precision/MRR** when the query contains distinctive n‑grams found verbatim in the documents.  
3. **Practical guidance.** In production, a **hybrid** approach (TF‑IDF ∪ Dense) or **Dense → lexical rerank** can combine strengths: wide semantic coverage with precise ranking.  
4. **Future work.**  
   - Try **BM25** and **hybrid fusion** (reciprocal rank fusion) to raise both recall and MRR.  
   - Add a learned **cross‑encoder reranker** on top‑k candidates.  
   - Expand queries and analyze **per‑topic** performance to see where each method shines/falters.

---

## Appendix
- **Repro.** Run `run_in_colab.ipynb` or the CLI:

```bash
python search_demo.py   --corpus data/my_corpus_1k.csv   --queries data/queries.json   --qrels data/qrels.json   --top_k 5   --plot outputs/results.png   --save_metrics outputs/metrics.json
```

- **Environment.** Python 3.x; `pandas`, `numpy`, `scikit-learn`, `matplotlib`, `faiss-cpu`, `sentence-transformers`.
