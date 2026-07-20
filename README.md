                                  NHS LOS Predictive Tool Evaluation

## Overview

This notebook analyses free-text survey responses from the WWL (Wrightington, Wigan and Leigh) NHS Foundation Trust staff survey on the Length of Stay (LOS) predictive tool, alongside the quantitative Likert-scale survey data. It compares four sentiment analysis models and three hyperparameter optimisation techniques applied to the best-performing model, in order to establish a robust, defensible sentiment classification of staff free-text responses (barriers, benefits, and other open-ended fields).

**Dataset:** 1,482 survey responses (quantitative Likert items) and 2,110 free-text entries (qualitative barriers/benefits comments).

## What the Notebook Does

The notebook runs in two broad phases:

### Phase 1 — Quantitative & Qualitative Exploratory Analysis
- Descriptive statistics for all Likert-scale questions (mean, median, std, etc.)
- Awareness and adoption breakdowns (e.g. tool awareness vs. actual usage)
- Respondent demographics (role, department) visualised with matplotlib/seaborn
- Perceived usefulness analysis (H1–H3 items) by staff role
- **Kruskal-Wallis tests** to check whether scores differ significantly across roles
- Word clouds and text cleaning of open-text barrier/benefit responses
- **BERTopic** topic modelling to discover themes in "biggest barriers" responses
- **Zero-shot classification** (BART/DeBERTa) of free text into predefined categories

### Phase 2 — Sentiment Model Comparison
Four sentiment models are run on the same free-text data and compared side by side:
1. **RoBERTa** (2020) — transformer-based sentiment classifier
2. **VADER** — rule-based lexicon sentiment scorer (nltk)
3. **TextBlob** — polarity-based sentiment scorer
4. **ModernBERT** (2024) — newer transformer-based classifier

Results are compared as grouped bar charts (overall and by question source, e.g. "Barriers"), with a final summary table across all four models.

### Phase 3 — Hyperparameter Tuning on ModernBERT
Because ModernBERT was selected as the primary model, its inference hyperparameters (`max_length`, `confidence_threshold`, `batch_size`) are tuned using three different search strategies, each run as an independent experiment:

1. **Grid Search** — exhaustive test of 9 fixed parameter configurations
2. **Bayesian Optimisation (Optuna)** — 10 trials, using parameter-importance analysis and "best-so-far" convergence tracking
3. **Random Search** — 10 randomly sampled configurations

The notebook concludes with a combined comparison of the best configuration found by each of the three techniques (bar charts + summary table), evaluating which technique/configuration best surfaces negative sentiment in the "Barriers" responses.

## Structure at a Glance

| Section | Cells (approx.) | Purpose |
|---|---|---|
| Descriptive stats & demographics | 1–7 | Likert stats, role breakdowns, Kruskal-Wallis |
| NLP setup & topic modelling | 8–19 | Word clouds, RoBERTa sentiment, BERTopic, zero-shot classification |
| VADER vs TextBlob vs RoBERTa | 20–27 | Three-way sentiment model comparison |
| ModernBERT sentiment | 28–35 | Adds ModernBERT as a 4th model, full 4-model comparison |
| ModernBERT hyperparameter configs | 36–44 | Manual comparison of 5 hand-picked config combinations |
| Reload / checkpoint | 45–50 | Reloads saved labels (`text_df_all_labels.csv`) to resume work |
| Grid Search | 51–59 | 9-configuration exhaustive grid search |
| Optuna (Bayesian) | 60–67 | 10-trial Bayesian optimisation |
| Random Search | 68–80 | 10-configuration random search |
| Final comparison | 81–87 | Best result per technique, final verdict |

## Requirements

Install via the first notebook cell, or manually:

```bash
pip install pandas numpy matplotlib seaborn scipy plotly wordcloud nltk
pip install torch --index-url https://download.pytorch.org/whl/cpu
pip install transformers sentence-transformers bertopic umap-learn hdbscan
pip install textblob vaderSentiment optuna
```

NLTK resources needed: `stopwords`, `punkt`, `vader_lexicon` (downloaded within the notebook).

## Data Files

- **Input:** the survey dataset (Likert columns such as `B1_awareness_clean`, `A1_role_clean`, `H1_clean`–`H3_clean`; free-text columns such as `K2_open_biggest_barriers`) — expected to already be loaded as `df` before Section 1 runs.
- **Intermediate/checkpoint output:** `text_df_all_labels.csv` — saves all four models' sentiment labels so the notebook can be resumed without re-running the transformer models (which take 10–15 minutes each on CPU).
- **Other output:** `random_search_results.csv` — results of the random search hyperparameter sweep.

## Runtime Notes

- Transformer models (RoBERTa, ModernBERT, BART/DeBERTa zero-shot, sentence-transformers for BERTopic) are CPU-heavy. Each full pass over the 2,110 free-text responses can take **10–15 minutes**.
- The Grid Search, Optuna, and Random Search sections each run **9–10 full model passes**, so each of these three sections can take **~100–150 minutes** on a CPU-only machine. The notebook explicitly warns not to restart the kernel mid-run.
- Because of this runtime cost, the notebook checkpoints sentiment labels to CSV (`text_df_all_labels.csv`) so later sections can reload results instead of recomputing them.

## Key Outputs

- Descriptive and inferential statistics (Kruskal-Wallis) on Likert survey items by staff role
- Word clouds and discovered topics (BERTopic) for barrier responses
- Zero-shot category classification of open-text responses
- A 4-model sentiment comparison table/chart (RoBERTa, VADER, TextBlob, ModernBERT)
- A 3-technique hyperparameter comparison table/chart (Grid Search, Optuna, Random Search) identifying the best-performing ModernBERT configuration for surfacing negative sentiment in barrier responses

## Suggested Use

Run top to bottom on first use. On subsequent runs, you can skip directly to Cell ~50 (`text_df = pd.read_csv('text_df_all_labels.csv')`) if `text_df_all_labels.csv` already exists, to avoid re-running the transformer models.
