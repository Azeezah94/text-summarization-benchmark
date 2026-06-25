# Automatic Text Summarization: A Transformer Model Benchmark
**Author:** Azeezat Akinola | [LinkedIn](https://www.linkedin.com/in/azeezat-akinola-710b73113) | [Portfolio](https://Azeezah94.github.io)

Fine-tunes and benchmarks four leading transformer models **Pegasus**, **BART**, **T5**, and **FLAN-T5** on Amazon review summarization, scored with ROUGE and statistical testing.

## Methodology
Matches the published thesis methodology exactly:
- Fine-tune each model on Amazon review data (review text -> review title/summary)
- Score generated summaries with **ROUGE-1, ROUGE-2, ROUGE-L**
- Run **paired t-tests** across models to determine statistically meaningful differences
- Evaluate for fluency and structural coherence

## Results (from thesis)
| Finding | Result |
|---|---|
| Top performer | **Pegasus** — excelled in fluency and structural coherence |
| Evaluation metric | ROUGE (1, 2, L) |
| Statistical method | Paired t-tests |

## Tech Stack
Python, HuggingFace Transformers, Pegasus, BART, T5, FLAN-T5, ROUGE, SciPy, Datasets

## Setup
```bash
git clone https://github.com/Azeezah94/text-summarization-benchmark
cd text-summarization-benchmark
pip install -r requirements.txt
python src/summarization_benchmark.py
```

## Publication
Part of: *"Evaluating Multimodal AI Systems: A Comparative Analysis of Large Language Model-Based Models for Text, Image, and Video Generation"*
M.S. Thesis, Georgia Southern University, 2025 — [Read the full thesis](https://digitalcommons.georgiasouthern.edu/etd/2944/)
