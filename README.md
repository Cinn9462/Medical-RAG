# Confidence-Gated Candidate-Specific Retrieval for Medical MCQA

This repository contains the experiments and analysis behind the paper [Confidence-Gated Candidate-Specific Retrieval for Medical Multiple-Choice Question Answering](Confidence-Gated%20Candidate-Specific%20Retrieval%20for%20Medical%20Multiple-Choice%20Question%20Answering.pdf).

The project studies whether retrieval can be used selectively for medical multiple-choice questions. Rather than retrieve evidence only for the question or retrieve again for every question, the proposed method first scores answer choices, detects when several choices remain plausible, and retrieves targeted evidence for those candidate choices before making a final prediction.

## Experiment overview

All answer choices are scored using the next-token logits of the same generative language model. The logits for the valid answer letters are converted to a softmax distribution. Let `p_max` be the largest choice probability and `T` the confidence threshold. The implementation treats a choice as a candidate when:

```text
p(choice) >= p_max - T
```

If exactly one choice meets this condition, the model keeps the initial answer without a second retrieval. If multiple choices meet it, the model generates a focused medical search query for each candidate choice, retrieves additional passages, and makes a second answer prediction with the new context. Thus `T` controls how often the additional retrieval step runs: increasing it generally admits more candidates and triggers more retrieval.

For retrieval, the code embeds the question or candidate-specific query with `BAAI/bge-base-en-v1.5`, searches a Milvus collection using cosine similarity, and optionally reranks retrieved passages with `BAAI/bge-reranker-v2-m3`. The main experiment uses the Llama model described below. The implementation uses CUDA and bfloat16.

## Datasets

The paper's final evaluation set contains approximately 1.6k questions across two multiple-choice medical QA datasets. A separate 1k-question training/calibration set is used for the ablation sweep and threshold selection; it is not the evaluation-set size.

| Dataset | Prepared questions in this repository | Format |
| --- | ---: | --- |
| MedQA (USMLE) | 1,273 | 5 answer choices |
| MedMCQA | 1,000 | 4 answer choices |

`question-set/medqa_raw.jsonl` contains the MedQA source records used to prepare `medqa_1273.csv`. `medmcqa_1000.csv` is a 1,000-question subject-stratified sample from the MedMCQA validation split. The scripts also produce a 20-question MedMCQA sample for quick experiments. See the source dataset terms before reuse or redistribution.

The prepared files contain 1,272 MedQA and 1,000 MedMCQA questions. The paper's reported evaluation uses the 1,672-question evaluation split, while the separate 1,000-question set is used to train/calibrate the ablation sweep and select the threshold. 

## Models and retrieval components

| Component | Role |
| --- | --- |
| `meta-llama/Llama-3.1-8B-Instruct` | Generates answer-choice logits; also generates candidate-specific retrieval queries in the gated method. |
| `BAAI/bge-base-en-v1.5` | Embeds question and candidate-specific queries for vector search. |
| `BAAI/bge-reranker-v2-m3` | Reranks the initial retrieved passages; scripts retrieve up to 15 candidates and retain the top 5 when reranking is enabled. |
| Milvus | Stores and searches the medical text collection. The experiment scripts expect a collection named `MedRAG_combined_collection`. |

## Results

The table below reports the paper's final comparison on the approximately 1,672-question evaluation set. Latency is the reported per-question latency.

| Model configuration | Overall accuracy | MedMCQA accuracy | MedQA accuracy | Per-question latency |
| --- | ---: | ---: | ---: | ---: |
| Baseline | 55.21% | 54.73% | 55.56% | 0.049 s |
| Baseline CoT | **63.48%** | 59.14% | **66.60%** | 6.331 s |
| RAG baseline | 59.09% | 61.09% | 57.66% | 0.161 s |
| RAG reranker | 59.75% | 61.80% | 58.27% | 0.363 s |
| RAG CoT | 62.66% | 62.32% | 62.90% | 6.696 s |
| RAG HyDE | 61.88% | 59.46% | 63.92% | 2.561 s |
| HyDE Gate | 62.56% | 61.087% | 63.62% | 0.977 s |
| RAG dynamic (`τ = 0.4`) | 62.86% | 62.80% | 63.08% | 1.368 s |
| Dynamic NoGate | 63.40% | **63.81%** | 63.10% | 3.66 s |

The table values are from the paper's final evaluation and are distinct from the training/calibration sweep saved under `results-and-visualizations/iterative_calibration/`. The sweep explores threshold-dependent accuracy and latency using the 1000 training/calibration set; it supports choosing `τ = 0.4` for the reported RAG dynamic configuration.

Other experiment result folders include `baseline`, `baseline_cot`, `rag_baseline`, `rag_cot`, `rag_rr`, `hyde_rag`, `iterative_0.4`, and `iterative_1.0`. Their CSVs store per-question probabilities, answer labels, timing, error flags, and sometimes retrieved passage metadata. The ready-made `iterative_calibration/accuracy_vs_threshold_with_time.png` visualizes a calibration sweep. The results scripts were written at different stages and some target legacy paths, so inspect their configured paths before rerunning.

## Repository layout and scripts

### `models/`

| Script | Purpose |
| --- | --- |
| `baseline.py` | Scores MedQA and MedMCQA choices without retrieval using next-token answer-letter logits. |
| `baseline_cot.py` | Generates a chain-of-thought style answer, then extracts answer-choice scores from generation. |
| `rag_baseline.py` | Retrieves passages for the question and choices, optionally reranks them, then scores choices with context. |
| `rag_cot.py` | RAG variant that generates reasoning before extracting the final choice. |
| `hyde_style_rag.py` | Generates a hypothetical medical passage (HyDE), retrieves/reranks against it, and answers using retrieved context. |
| `rag_iterative.py` | Confidence-gated candidate-specific retrieval and threshold evaluation; generates a query for each near-top candidate and retries with added context when multiple candidates qualify. |
| `query_test.py` | Retrieval and generation experimentation across separate Milvus collections; includes an 8-bit model configuration. |

### `question-set/`

| File | Purpose |
| --- | --- |
| `medqa_raw.jsonl` | Raw MedQA records used to create the prepared MedQA CSV. |
| `medqa_1273.csv` | Prepared 5-choice MedQA evaluation set. |
| `medmcqa_1000.csv` | Prepared subject-stratified MedMCQA sample. |
| `medmcqa_20.csv` | Small MedMCQA sample for quick checks. |
| `split generation.py` | Converts MedQA JSONL and samples MedMCQA from Hugging Face Datasets. It currently uses sampling without a fixed random seed, so regeneration may produce a different sample. |

### `milvus/`

| Script | Purpose |
| --- | --- |
| `milvus-insertion.py` | Creates separate textbook, StatPearls, and PubMed collections and inserts embeddings from MedRAG Hugging Face datasets plus processed StatPearls chunks. |
| `milvus-check.py` | Loads and queries the configured combined Milvus collection for retrieval inspection. |
| `statpearls/processing.py` | Parses StatPearls XML into text chunks for ingestion. |

The ingestion script creates three source-specific collections; it does not by itself create `MedRAG_combined_collection`, which the main RAG scripts expect. The combined index must be prepared separately or the scripts' collection settings must be aligned.

### `results-and-visualizations/`

This directory contains saved experiment CSVs/plots as well as scripts for summarization, confidence analysis, calibration, filtering, and combining gated/always-retrieve outputs:

| Script | Purpose |
| --- | --- |
| `summary.py` | Summarizes a configured pair of MedQA and MedMCQA result files. |
| `visualizations.py` | Builds confidence, answer-distribution, source, and top-k plots from configured result CSVs. |
| `filter.py` | Aggregates result folders and reports accuracy, latency, and error rate, with optional exclusion IDs. |
| `iterative_analysis.py` | Compares iterative and non-iterative outcomes using configured baseline, dynamic, and calibration paths. |
| `hyde_gated/combine.py` | Selects the initial or always-retrieved prediction according to a probability-margin threshold and reports a sweep. |
| `iterative_calibration/calibration curve.py` | Plots accuracy and time across thresholds from the iterative calibration outputs. |

Several analysis scripts contain stale paths such as `iterative_ablation` or `rag_rr_baseline`, which are absent from the checked-in directory tree. The available outputs currently live under `iterative_calibration`, `rag_rr`, and the other named experiment folders. Adjust paths before invoking affected scripts.

## Setup and running

Requirements:

- Python 3.10 or newer.
- NVIDIA GPU/CUDA with support for the scripts' CUDA and bfloat16 settings, and enough VRAM for the 8B model plus embedding and reranker models.
- Milvus running at `http://localhost:19530` for RAG and indexing scripts.
- Hugging Face access to `meta-llama/Llama-3.1-8B-Instruct` (including acceptance of its model terms).
- Internet on first run to download model weights and datasets.

Create a virtual environment and install dependencies:

```bash
python -m venv .venv
# Windows PowerShell
.\.venv\Scripts\Activate.ps1
# macOS/Linux
source .venv/bin/activate
python -m pip install --upgrade pip
pip install -r requirements.txt
```

Install a PyTorch wheel appropriate to your CUDA setup if the default pip package is unsuitable. Authenticate with Hugging Face using `hf auth login`. Run scripts from the repository root because they use relative paths. Model scripts execute experiments directly and include hard-coded settings; review the script before starting a long run.