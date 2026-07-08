# Testing a Low-Dimensional Alternative to Bag-of-Words Text Classification

**What this project shows**: how to design, build, and honestly evaluate a compact feature set for text classification, instead of defaulting to a high-dimensional bag-of-words representation, and how to read a negative result correctly instead of dressing it up.

The task here is a health-text classification problem (distinguishing blog posts by author group using only 5 engineered features: lexical diversity + sentiment). But the underlying skill is general: **any team working with unstructured text,  support tickets, product reviews, survey open-ends, clinical notes, user research transcripts, eventually has to choose between an interpretable, low-dimensional feature set and a high-dimensional one, and needs to know how to tell whether the compact version is actually good enough.**

## Why this matters in an industry context

| What this project does | Where this shows up in industry |
|---|---|
| Builds a small, interpretable feature set (5 numbers per document) instead of a large bag-of-words matrix | Feature engineering for text classifiers where interpretability matters, compliance review, customer support routing, moderation, not just raw accuracy |
| Runs a proper cross-validated hyperparameter search and reports the full curve, not just the best number | Standard practice for any classifier going into production: tuning transparently instead of cherry-picking a result |
| Diagnoses *why* a model plateaus (a flat F1 curve across six orders of magnitude of regularization) instead of just reporting the final metric | The analytical habit that catches "the model isn't the problem, the features are" before a team burns weeks tuning the wrong thing |
| States a negative result plainly and names the likely confound (author overlap between train/test folds) instead of overstating the model's performance | The kind of rigor that keeps a report credible when it reaches a stakeholder who will act on it |

## What's inside

- **Feature engineering for text**: lexical diversity (MATTR) and sentiment (VADER) as a low-dimensional alternative to TF-IDF
- **Cross-validated hyperparameter search**: logistic regression with grid search over regularization strength, evaluated with F1-macro under 5-fold CV
- **Model diagnostics**: confusion matrix, per-class precision/recall/F1, and a clear read on what a flat tuning curve actually means
- **Honest limitations analysis**: naming the specific methodological risk (author leakage across CV folds) that would need to be fixed before trusting the reported accuracy

## Research question

Can a low-dimensional feature set, lexical diversity plus sentiment, meaningfully separate two groups of text authors, as a lighter-weight alternative to a high-dimensional bag-of-words representation?

## Data

- **Source**: blog corpus from Masrani, Murray, Field & Carenini (2017), BioNLP Workshop — posts written by individuals with Alzheimer's disease (class 1, N=2,278) and by family members as controls (class 2, N=1,373)
- **Not included in this repository**: the corpus contains real, identifiable health-related text. See [`data/README.md`](data/README.md) for how to source it and where to place it locally.
- Each document is parsed from `<doc date="...">...</doc>`-delimited entries

## Method

1. **Parsing**: reads the tagged corpus into one record per document (date, text, label)
2. **Feature extraction**:
   - MATTR (Moving-Average Type-Token Ratio, 50-token window) on lowercased, punctuation-stripped text — a length-independent measure of vocabulary diversity
   - VADER sentiment scores (negative, neutral, positive, compound) on the original, unprocessed text, since casing and punctuation carry sentiment information
3. **Model**: logistic regression (`scikit-learn`), L2-regularized, `liblinear` solver, `class_weight="balanced"` to handle the class imbalance
4. **Tuning**: grid search over `C` (0.001 to 1000), 5-fold cross-validation, F1-macro as the selection metric
5. **Evaluation**: confusion matrix and classification report on out-of-fold predictions at the best `C`

## Key results

| C | F1-macro |
|---|---|
| 0.001 – 10 | 0.50 – 0.52 (essentially flat) |
| 100 – 1000 | 0.52 (no further change) |

- Best F1-macro: **0.52** at C=10 — close to chance for a two-class problem
- Precision and recall trade off in opposite directions between the two classes (AD: precision 0.68 / recall 0.45; HC: precision 0.42 / recall 0.65), consistent with the model redistributing errors rather than finding a genuine separating signal
- The flat curve across six orders of magnitude of regularization is the most useful finding here: it rules out under/over-regularization as the explanation and points squarely at the feature set itself

Full discussion, references, and the reasoning behind each design choice are in [`docs/report.md`](docs/report.md).

## Repository structure

```
.
├── lexical_sentiment_classification.ipynb   # full pipeline: parsing, feature extraction, model, evaluation
├── data/
│   ├── README.md                             # data sourcing notes (raw corpus not included)
│   └── raw/                                  # place class1.txt / class2.txt here (gitignored)
├── docs/
│   └── report.md                             # full write-up with tables and references
├── requirements.txt
└── README.md
```

## Reproducing the analysis

```bash
pip install -r requirements.txt
# obtain the corpus (see data/README.md) and place it under data/raw/
jupyter notebook lexical_sentiment_classification.ipynb
```

## Limitations

- **Possible author leakage across cross-validation folds.** If the same author contributes multiple posts, and posts from one author end up in both training and test folds, the model may be learning author-specific style rather than generalizing to new authors. This is the single most important caveat on the reported F1-macro, and grouped (author-level) cross-validation is the fix to apply before treating this result as final.
- **Small feature set, by design.** Five features is intentionally minimal; the negative result here shouldn't be read as "lexical diversity and sentiment don't matter for this task," but as "on their own, at this scale, they aren't sufficient."
- **No comparison against a TF-IDF baseline in this notebook.** The write-up references TF-IDF's known behavior on this type of task, but a direct side-by-side run is a natural next step rather than something assumed.
