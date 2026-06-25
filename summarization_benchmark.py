"""
Automatic Text Summarization: A Transformer Model Benchmark
Author: Azeezat Akinola
Thesis: "Evaluating Multimodal AI Systems" — Georgia Southern University, 2025
        https://digitalcommons.georgiasouthern.edu/etd/2944/

Fine-tunes and evaluates four transformer models — Pegasus, BART, T5, and
FLAN-T5 — on Amazon review summarization, scored via ROUGE with paired
t-tests for statistical comparison.
"""

import numpy as np
import pandas as pd
from scipy import stats
from datasets import load_dataset
from transformers import (
    AutoTokenizer, AutoModelForSeq2SeqLM,
    Seq2SeqTrainer, Seq2SeqTrainingArguments,
    DataCollatorForSeq2Seq,
)
from rouge_score import rouge_scorer


MODEL_CHECKPOINTS = {
    "Pegasus":   "google/pegasus-xsum",
    "BART":      "facebook/bart-large-cnn",
    "T5":        "t5-small",
    "FLAN-T5":   "google/flan-t5-base",
}


# ── Data Loading ───────────────────────────────────────────────────────────────
def load_amazon_reviews(sample_size: int = 1000):
    """
    Loads Amazon review data for summarization fine-tuning.
    Uses the amazon_reviews_multi / amazon_polarity dataset structure —
    reviews as source text, review titles/summaries as target.
    """
    dataset = load_dataset("amazon_polarity", split="train")
    dataset = dataset.select(range(sample_size))
    return dataset.map(lambda x: {
        "source": x["content"],
        "target": x["title"],
    })


# ── Fine-tuning ────────────────────────────────────────────────────────────────
def fine_tune_model(model_key: str, train_dataset, output_dir: str,
                    epochs: int = 3, batch_size: int = 8):
    """Fine-tunes a given transformer model on the summarization dataset."""
    checkpoint = MODEL_CHECKPOINTS[model_key]
    tokenizer = AutoTokenizer.from_pretrained(checkpoint)
    model = AutoModelForSeq2SeqLM.from_pretrained(checkpoint)

    def preprocess(batch):
        inputs = tokenizer(batch["source"], max_length=512, truncation=True, padding="max_length")
        targets = tokenizer(batch["target"], max_length=64, truncation=True, padding="max_length")
        inputs["labels"] = targets["input_ids"]
        return inputs

    tokenized = train_dataset.map(preprocess, batched=True)

    training_args = Seq2SeqTrainingArguments(
        output_dir=output_dir,
        per_device_train_batch_size=batch_size,
        num_train_epochs=epochs,
        save_strategy="epoch",
        predict_with_generate=True,
        logging_steps=50,
    )

    trainer = Seq2SeqTrainer(
        model=model, args=training_args,
        train_dataset=tokenized,
        data_collator=DataCollatorForSeq2Seq(tokenizer, model=model),
        tokenizer=tokenizer,
    )
    trainer.train()
    return model, tokenizer


# ── ROUGE Evaluation ───────────────────────────────────────────────────────────
class ROUGEEvaluator:
    def __init__(self):
        self.scorer = rouge_scorer.RougeScorer(
            ["rouge1", "rouge2", "rougeL"], use_stemmer=True
        )

    def evaluate(self, predictions: list[str], references: list[str]) -> pd.DataFrame:
        rows = []
        for pred, ref in zip(predictions, references):
            scores = self.scorer.score(ref, pred)
            rows.append({
                "rouge1": scores["rouge1"].fmeasure,
                "rouge2": scores["rouge2"].fmeasure,
                "rougeL": scores["rougeL"].fmeasure,
            })
        return pd.DataFrame(rows)


# ── Benchmark Runner ───────────────────────────────────────────────────────────
def run_benchmark(test_texts: list[str], reference_summaries: list[str],
                  models: list[str] = None) -> pd.DataFrame:
    """Generates summaries from each model and scores them against references."""
    models = models or list(MODEL_CHECKPOINTS.keys())
    evaluator = ROUGEEvaluator()
    all_results = []

    for model_key in models:
        checkpoint = MODEL_CHECKPOINTS[model_key]
        tokenizer = AutoTokenizer.from_pretrained(checkpoint)
        model = AutoModelForSeq2SeqLM.from_pretrained(checkpoint)

        predictions = []
        for text in test_texts:
            inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=512)
            summary_ids = model.generate(**inputs, max_length=64, num_beams=4)
            predictions.append(tokenizer.decode(summary_ids[0], skip_special_tokens=True))

        scores = evaluator.evaluate(predictions, reference_summaries)
        scores["model"] = model_key
        all_results.append(scores)

    return pd.concat(all_results, ignore_index=True)


# ── Statistical Comparison (matches thesis: paired t-tests) ───────────────────
def statistical_comparison(df: pd.DataFrame) -> dict:
    """Paired t-tests across models on ROUGE-L, identifying the top performer."""
    models = df["model"].unique()
    summary = df.groupby("model")[["rouge1", "rouge2", "rougeL"]].mean().round(4)

    pairwise = {}
    for i in range(len(models)):
        for j in range(i + 1, len(models)):
            g1 = df[df["model"] == models[i]]["rougeL"].values
            g2 = df[df["model"] == models[j]]["rougeL"].values
            n = min(len(g1), len(g2))
            if n > 1:
                t_stat, p_val = stats.ttest_rel(g1[:n], g2[:n])
                pairwise[f"{models[i]} vs {models[j]}"] = {
                    "t_stat": round(t_stat, 4),
                    "p_value": round(p_val, 4),
                    "significant": p_val < 0.05,
                }

    top_model = summary["rougeL"].idxmax()
    return {
        "summary_by_model": summary.to_dict(),
        "top_performer": top_model,
        "pairwise_t_tests": pairwise,
    }


# ── Demo ──────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    sample_texts = [
        "This product exceeded my expectations. The build quality is excellent and "
        "it works exactly as described. Shipping was fast and packaging was secure.",
        "Terrible experience. The item arrived damaged and customer service was unhelpful. "
        "Would not recommend this seller to anyone looking for quality products.",
    ]
    sample_references = [
        "Excellent quality product, fast shipping",
        "Damaged item, poor customer service",
    ]

    print("Running ROUGE benchmark across Pegasus, BART, T5, FLAN-T5...")
    results = run_benchmark(sample_texts, sample_references)
    stats_summary = statistical_comparison(results)

    print("\n=== Mean ROUGE Scores by Model ===")
    print(pd.DataFrame(stats_summary["summary_by_model"]))
    print(f"\nTop performer: {stats_summary['top_performer']}")
    print("\n=== Paired t-tests (ROUGE-L) ===")
    for comparison, result in stats_summary["pairwise_t_tests"].items():
        sig = "significant" if result["significant"] else "not significant"
        print(f"  {comparison}: t={result['t_stat']}, p={result['p_value']} ({sig})")
