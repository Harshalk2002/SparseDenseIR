# Assignment 1 — Sparse (TF‑IDF) vs Dense (Sentence‑Transformers) Retrieval

**Submit this folder as a ZIP** or upload `run_in_colab.ipynb` to Colab and run all cells.

## What’s included
- `run_in_colab.ipynb`: One‑click Colab notebook that installs deps, builds TF‑IDF + FAISS indexes, runs 10 eval queries, and plots results.
- `search_demo.py`: CLI script that does the same locally.
- `data/my_corpus_1k.csv`: Synthetic 1,000‑doc corpus spanning 10 topics (stable for grading).
- `data/queries.json`: 10 natural queries.
- `data/qrels.json`: Ground‑truth relevant doc ids per query (for Recall@K / MRR).
- `report_template.md`: 3–4 page outline you can fill in and export to PDF.
- `requirements.txt`: Local install list.
- `outputs/`: Results and plots are written here when you run either the notebook or the script.

## Quick Start (Colab recommended)
1. Open **`run_in_colab.ipynb`** in Google Colab.
2. Runtime → Run all. This will:
   - `pip install` requirements (scikit‑learn, sentence‑transformers, faiss‑cpu, pandas, matplotlib, numpy).
   - Build TF‑IDF and Dense (all‑MiniLM‑L6‑v2) indexes from `data/my_corpus_1k.csv`.
   - Evaluate the 10 queries from `data/queries.json` using `data/qrels.json`.
   - Save metrics and plots to `outputs/` (e.g., `results.png`, `metrics.json`).

## Local Run (Python 3.9+ suggested)
```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

python search_demo.py \
  --corpus data/my_corpus_1k.csv \
  --queries data/queries.json \
  --qrels data/qrels.json \
  --top_k 5 \
  --plot outputs/results.png \
  --save_metrics outputs/metrics.json
```

## Deliverables
- **Notebook or scripts with outputs** (this repo provides both)
- **3–4 page report** (use `report_template.md` → export to PDF)
- Plots + tables are generated to `outputs/`

> Tip: You can replace `data/my_corpus_1k.csv` with your own corpus; the code will still work.
