# 20252R0136DATA30400 — DATA304 Final Project

This repository contains my solution code for the **DATA304 (Big Data Analysis) Final Project**.

## Project Setup / Reproducibility

**All code is runnable by copying the contents of this repository into `./project_release/`** (the project-provided directory) and executing from there.

The easiest way to reproduce the full pipeline is:

- Run **`reproducibility_test.ipynb`**  
  → It executes the main scripts in order and produces the corresponding outputs/submissions.

> Note: Scores below are **Kaggle public leaderboard scores** obtained from the generated submission files.

---

## Results Summary (Kaggle Scores)

| Method / Submission | Script | Score |
|---|---|---:|
| Dummy baseline | `dummy_baseline.ipynb` | 0.00436 |
| TF-IDF baseline | `tfidf_baseline_fast.py` | 0.21251 |
| Self-training (pseudo-labeling, threshold = 0.5) | `selftrain_baseline.py` | **0.21334** |
| GNN v1 | `gnn_baseline.py` | 0.10250 |
| GNN v2 | `gnn_baseline_2.py` | 0.14832 |
| GNN v3 | `gnn_baseline_3.py` | 0.12793 |
| GNN v4 (hidden=256, epochs=20) | `gnn_baseline_4.py` | 0.16737 |

---

## File Guide

### Notebooks
- **`dummy_baseline.ipynb`**  
  Project-provided dummy baseline.  
  **Score:** 0.00436

- **`reproducibility_test.ipynb`**  
  A guided notebook to reproduce the full workflow by running the scripts below in sequence.

### Core Scripts
- **`silver_label_generation_sagemaker.py`**  
  Generates **silver labels** using class-related keywords and hierarchy information.

- **`tfidf_baseline_fast.py`**  
  TF-IDF + linear classifier baseline trained on silver labels.  
  **Score:** 0.21251

- **`selftrain_baseline.py`**  
  Self-training baseline with pseudo-labeling.  
  Main run uses **confidence threshold = 0.5**.  
  **Score:** 0.21334

### GNN Experiments
- **`gnn_baseline.py`** — GNN v1 (initial LabelGCN baseline)  
- **`gnn_baseline_2.py`** — GNN v2 (valid class coverage improvement)  
- **`gnn_baseline_3.py`** — GNN v3 (stability fixes + epochs increase)  
- **`gnn_baseline_4.py`** — GNN v4 (hidden dim / epochs scaled up)  
  Best GNN score: **0.16737**

### Case Study
- **`case_study_extraction.py`**  
  Extracts qualitative success/failure examples from test predictions for the report case study section.

---

## Suggested Run Order (Scripts)

If running manually (outside the notebook), a typical order is:

1. `silver_label_generation_sagemaker.py`  
2. `tfidf_baseline_fast.py`  
3. `selftrain_baseline.py`  
4. `gnn_baseline_4.py` (or other GNN variants)  
5. `case_study_extraction.py`

