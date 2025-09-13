#!/usr/bin/env python3
import argparse, json, sys, math
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Import sentence-transformers + faiss lazily (nice errors if missing)
def _lazy_import_dense():
    try:
        from sentence_transformers import SentenceTransformer
    except Exception as e:
        print("ERROR: sentence-transformers not installed. Install with:\n  pip install sentence-transformers", file=sys.stderr)
        raise
    try:
        import faiss
    except Exception as e:
        print("ERROR: faiss-cpu not installed. Install with:\n  pip install faiss-cpu", file=sys.stderr)
        raise
    return SentenceTransformer, faiss

def build_tfidf_index(texts):
    vect = TfidfVectorizer(ngram_range=(1,2), min_df=2)
    X = vect.fit_transform(texts)
    return vect, X

def tfidf_search(vect, X, query, top_k=5):
    qv = vect.transform([query])
    sims = cosine_similarity(qv, X).ravel()
    idx = np.argsort(-sims)[:top_k]
    return idx, sims[idx]

def build_dense_index(texts, model_name="sentence-transformers/all-MiniLM-L6-v2"):
    SentenceTransformer, faiss = _lazy_import_dense()
    model = SentenceTransformer(model_name)
    emb = model.encode(texts, batch_size=64, show_progress_bar=False, normalize_embeddings=True)
    dim = emb.shape[1]
    index = faiss.IndexFlatIP(dim)
    index.add(emb.astype(np.float32))
    return model, index, emb

def dense_search(model, index, query, top_k=5):
    import numpy as np
    q = model.encode([query], normalize_embeddings=True)
    D, I = index.search(q.astype(np.float32), top_k)
    return I[0], D[0]

def load_corpus(fp):
    df = pd.read_csv(fp)
    assert {"doc_id","text"}.issubset(df.columns), "CSV must have columns: doc_id,text"
    return df

def load_json(fp):
    with open(fp, "r", encoding="utf-8") as f:
        return json.load(f)

def eval_metrics(runs, qrels, k_list=(5,10)):
    # runs: dict qid -> list of doc_ids in rank order
    # qrels: dict qid -> set/list of relevant doc_ids
    out = {"Recall":{}, "MRR":{}}
    for k in k_list:
        recalls, rr = [], []
        for qid, ranked in runs.items():
            rel = set(qrels.get(qid, []))
            if not rel:
                continue
            topk = ranked[:k]
            hit = sum(1 for d in topk if d in rel)
            recalls.append(hit / max(1, len(rel)))
            # reciprocal rank up to k
            rrk = 0.0
            for i, d in enumerate(topk, 1):
                if d in rel:
                    rrk = 1.0 / i
                    break
            rr.append(rrk)
        out["Recall"][f"@{k}"] = float(np.mean(recalls)) if recalls else 0.0
        out["MRR"][f"@{k}"] = float(np.mean(rr)) if rr else 0.0
    return out

def plot_comparison(metrics_tfidf, metrics_dense, out_png):
    labels = ["Recall@5","Recall@10","MRR@5","MRR@10"]
    tvals = [metrics_tfidf["Recall"]["@5"], metrics_tfidf["Recall"]["@10"], metrics_tfidf["MRR"]["@5"], metrics_tfidf["MRR"]["@10"]]
    dvals = [metrics_dense["Recall"]["@5"], metrics_dense["Recall"]["@10"], metrics_dense["MRR"]["@5"], metrics_dense["MRR"]["@10"]]

    x = np.arange(len(labels))
    width = 0.35
    fig, ax = plt.subplots(figsize=(8,4.5))
    ax.bar(x - width/2, tvals, width, label="TF‑IDF")
    ax.bar(x + width/2, dvals, width, label="Dense")
    ax.set_ylabel("Score")
    ax.set_title("Sparse vs Dense Retrieval")
    ax.set_xticks(x, labels)
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_png, dpi=150)
    plt.close(fig)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--corpus", required=True, help="CSV with columns: doc_id,text")
    ap.add_argument("--queries", required=True, help="queries.json (list of {{qid, query}})")
    ap.add_argument("--qrels", required=True, help="qrels.json (qid -> [doc_id])")
    ap.add_argument("--top_k", type=int, default=5)
    ap.add_argument("--plot", default=None, help="Path to save comparison plot PNG")
    ap.add_argument("--save_metrics", default=None, help="Path to save metrics JSON")
    args = ap.parse_args()

    df = load_corpus(args.corpus)
    queries = load_json(args.queries)
    qrels = load_json(args.qrels)

    texts = df["text"].tolist()
    ids = df["doc_id"].tolist()

    # TF-IDF
    tfidf_vect, X = build_tfidf_index(texts)
    runs_tfidf = {}
    for item in queries:
        qid, q = item["qid"], item["query"]
        idx, _ = tfidf_search(tfidf_vect, X, q, top_k=max(args.top_k, 10))
        runs_tfidf[qid] = [ids[i] for i in idx]

    # Dense
    model, index, emb = build_dense_index(texts)
    runs_dense = {}
    for item in queries:
        qid, q = item["qid"], item["query"]
        idx, _ = dense_search(model, index, q, top_k=max(args.top_k, 10))
        runs_dense[qid] = [ids[i] for i in idx]

    # Evaluate
    m_tfidf = eval_metrics(runs_tfidf, qrels, k_list=(5,10))
    m_dense = eval_metrics(runs_dense, qrels, k_list=(5,10))

    print("TF‑IDF:", json.dumps(m_tfidf, indent=2))
    print("Dense :", json.dumps(m_dense, indent=2))

    if args.save_metrics:
        with open(args.save_metrics, "w", encoding="utf-8") as f:
            json.dump({"tfidf": m_tfidf, "dense": m_dense}, f, indent=2)

    if args.plot:
        plot_comparison(m_tfidf, m_dense, args.plot)
        print(f"Saved plot to {args.plot}")

if __name__ == "__main__":
    main()
