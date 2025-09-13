# Sparse vs Dense Retrieval — Report (3–4 pages)

**Course**: *Text_Analytics*
**Author**: *Harshal Kmable*  
**Date**: *September 13, 2025*

## 1. Dataset
- Source: Synthetic corpus (`data/my_corpus_1k.csv`) with 1,000 docs across 10 domains.
- Structure: `doc_id, topic, text`. Each topic has 100 docs with domain‑specific vocabulary.
- Queries: 10 natural queries (`data/queries.json`), each mapped to 10 relevant docs in `qrels.json`.

## 2. Methods
### 2.1 Sparse (TF‑IDF)
- Vectorizer: `TfidfVectorizer` (unigram+bigram), `min_df=2`.
- Similarity: cosine.
- Index: in‑memory (scikit‑learn).

### 2.2 Dense (Sentence‑Transformers + FAISS)
- Encoder: `sentence-transformers/all-MiniLM-L6-v2` (384‑dim).
- Index: FAISS `IndexFlatIP` on L2‑normalized vectors (cosine via dot).
- Batch encode corpus; build index once.

## 3. Experiments
- Evaluation queries (`data/queries.json`) and ground truth (`qrels.json`).
- Metrics: Recall@5, Recall@10, MRR@10.
- Report average across the 10 queries.
- Plot: bar chart comparing TF‑IDF vs Dense.

## 4. Results
- Include the numeric table and the plot (`outputs/results.png`).
- Briefly interpret which method performs better overall and by topic.

## 5. Error Analysis
- Provide 2–3 examples where TF‑IDF > Dense and where Dense > TF‑IDF.
- Hypothesize causes (lexical mismatch, synonymy, topic drift, etc.).

## 6. Takeaways
- When to prefer sparse vs dense.
- Cost/latency/memory tradeoffs.
- Any future work (hybrid retrieval, rerankers).

## Appendix
- Repro steps: point to `run_in_colab.ipynb` and CLI usage.
- Environment: package versions in `requirements.txt`.
