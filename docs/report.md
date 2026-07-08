# Lexical Diversity and Sentiment as Low-Dimensional Text Features for Alzheimer's Detection

## Introduction

A standard approach for text classification is TF-IDF: a high-dimensional bag-of-words representation that weighs each term by its frequency in a document and its rarity across the corpus. TF-IDF features can be single words or character n-grams; a common variant computes a TF-IDF score per class by concatenating all documents within it, retains the top-k features per class, and merges the two lists into a shared vocabulary (Masrani, Murray, Field, & Carenini, 2017).

This approach treats each feature independently and misses shared semantic content. It also produces a very wide feature matrix, which is expensive to interpret and prone to overfitting on smaller corpora.

This project tests a much lower-dimensional alternative: combining a measure of lexical diversity (MATTR) with a measure of sentiment (VADER), for a total of 5 features per document instead of thousands. The goal isn't to beat TF-IDF outright, but to test whether a compact, interpretable feature set can capture a meaningful signal on its own — and to report honestly on the result either way.

MATTR (Moving-Average Type-Token Ratio) quantifies vocabulary diversity in a way that's independent of document length. VADER is a lexicon-based sentiment classifier that returns four scores per document (negative, neutral, positive, compound) (Asthana et al., 2024).

## Data and features

The corpus is derived from Masrani et al. (2017): blog posts written either by patients with Alzheimer's disease (class 1) or by their relatives acting as controls (class 2). Class 1 contains 2,278 posts; class 2 contains 1,373 posts.

Each post is stored with a `<doc date="...">...</doc>` wrapper. A parser reads these tags and returns one record per document with its date, raw text, and class label.

Two feature families are computed per document:

- **MATTR**, using a standard 50-token window. Text is lowercased and stripped of punctuation before this step, since lexical diversity shouldn't depend on casing or punctuation.
- **VADER**, computed on the original, unprocessed text. Capitalization and punctuation carry sentiment information, so they're deliberately preserved for this step.

## Model and experimental setup

A logistic regression classifier was chosen for its simplicity given the small number of features. It was implemented with scikit-learn, with each feature standardized (fit only on the training fold, to avoid leakage) and `class_weight="balanced"` to compensate for the class imbalance in the corpus.

A grid search over the regularization parameter `C` (values: 0.001, 0.01, 0.1, 1, 10, 100, 1000) selected the setting that maximized F1-macro under 5-fold cross-validation. Lower `C` applies stronger regularization and favors simpler models; higher `C` lets the model fit the training data more closely, at the risk of hurting generalization. L2 regularization discourages any single feature from dominating the decision boundary. The solver was `liblinear` with up to 2000 iterations.

Performance was assessed with accuracy, precision, recall, and F1-score, with F1-macro used as the primary metric given the class imbalance.

## Results

The best regularization value was C=10, with an F1-macro of 0.52. Performance was essentially flat across C values spanning six orders of magnitude (0.50 to 0.52), which points to a ceiling in the feature set rather than a regularization problem.

| C | F1-macro (mean ± SD across folds) |
|---|---|
| 0.001 | 0.504 ± 0.065 |
| 0.01  | 0.507 ± 0.054 |
| 0.1   | 0.518 ± 0.064 |
| 1     | 0.521 ± 0.065 |
| 10    | 0.523 ± 0.066 |
| 100   | 0.523 ± 0.066 |
| 1000  | 0.523 ± 0.066 |

At C=10, per-class performance on the held-out test set:

| Class | Precision | Recall | F1-score | Support |
|---|---|---|---|---|
| AD (class 1) | 0.68 | 0.45 | 0.54 | 2,278 |
| HC (class 2) | 0.42 | 0.65 | 0.51 | 1,373 |
| **Accuracy** | | | **0.53** | 3,651 |
| Macro avg | 0.55 | 0.55 | 0.52 | 3,651 |
| Weighted avg | 0.58 | 0.53 | 0.53 | 3,651 |

Precision and recall trade off in opposite directions across the two classes, which is a sign the model is redistributing errors rather than exploiting a genuine separating signal.

## Discussion

Text classification for early screening is an increasingly active area, and dementia in particular is known to leave a detectable trace in written language: word frequency and vocabulary shift with disease progression, tracking the deterioration of the semantic memory system (Huff, Corkin, & Growdon, 1986). This lexical impoverishment tends to carry through to syntax too, producing shorter and more fragmented sentences. TF-IDF is sensitive to this, since low-frequency words show up more often in the reference corpus than in patients' text — but it does so at the cost of a very high-dimensional, largely uninterpretable feature space.

MATTR targets the same underlying phenomenon (vocabulary richness) with a single, length-independent number. VADER adds an orthogonal signal: dementia patients also show measurable changes in emotional expression (Chaudhary, Zhornitsky, Chao, van Dyck, & Li, 2022). Combining the two seemed like a reasonable, clinically-motivated way to compress the text-classification problem down to a handful of interpretable numbers.

The result: F1-macro landed just above chance for a two-class problem, and varying `C` across six orders of magnitude changed almost nothing. That flat curve is the most informative result in this analysis — it says the ceiling is in the features, not in how the model is tuned.

A few things are worth flagging as next steps rather than as reasons to discard the approach:

- **Feature set is compact but not necessarily sufficient.** MATTR and VADER are theoretically grounded but may need to be paired with additional signal — TF-IDF itself, other lexical diversity indices (Brunet's Index, Honoré's Statistic), topic coherence via LSA, or embeddings (GloVe, or contextual embeddings from transformer models).
- **A richer emotional model could help.** VADER's four scores are a coarse summary; a framework like EmoAtlas, built on Plutchik's emotion model, offers a more structured and psychologically grounded alternative (Semeraro et al., 2025).
- **Non-linear classifiers become worth testing once the feature set grows.** A linear boundary is a reasonable starting point for 5 features, but is unlikely to capture non-linear interactions if the feature set expands.
- **Author overlap between train and test folds is a real risk with this corpus.** If the same author appears in both the training and test splits, cross-validation folds can leak identity-specific writing style rather than testing generalization to unseen individuals. Grouped cross-validation (splitting by author, not by post) is the correct fix and should be treated as a prerequisite for trusting any accuracy number from this corpus, including the one reported here.

## Conclusion

Lexical diversity and sentiment are a reasonable, clinically-motivated basis for compressing a text classification problem down to a handful of interpretable features, and doing so revealed a clear result: on their own, without author-level cross-validation controls, they land close to chance for this task. That's a useful finding in itself — it rules out a hypothesis cleanly rather than leaving it ambiguous, and it points to where the actual signal likely lives (higher-dimensional lexical features, richer emotion models, or non-linear interactions between them).

## References

Asthana, P., Barnwal, M., Yadav, A., Aggrawal, M., & Goel, M. (2024). VADER: A lightweight and effective approach for sentiment analysis. In *2024 2nd International Conference on Advances in Computation, Communication and Information Technology (ICAICCIT)* (Vol. 1, pp. 687-692).

Chaudhary, S., Zhornitsky, S., Chao, H. H., van Dyck, C. H., & Li, C.-S. R. (2022). Emotion processing dysfunction in Alzheimer's disease: An overview of behavioral findings, systems neural correlates, and underlying neural biology. *American Journal of Alzheimer's Disease & Other Dementias*, 37.

Huff, F. J., Corkin, S., & Growdon, J. H. (1986). Semantic impairment and anomia in Alzheimer's disease. *Brain and Language*, 28(2), 235-249.

Masrani, V., Murray, G., Field, T. S., & Carenini, G. (2017). Detecting dementia through retrospective analysis of routine blog posts by bloggers with dementia. In *Proceedings of the 16th BioNLP Workshop* (pp. 232-237).

Sakai, H., & Lam, S. S. (2026). Large language models for health care text classification: Systematic review. *JMIR AI*, 5(1), e79202.

Semeraro, A., Vilella, S., Improta, R., De Duro, E. S., Mohammad, S. M., Ruffo, G., & Stella, M. (2025). EmoAtlas: An emotional network analyzer of texts that merges psychological lexicons, artificial intelligence, and network science. *Behavior Research Methods*, 57(2), 77.

Shakeri, A., & Farmanbar, M. (2025). Natural language processing in Alzheimer's disease research: Systematic review of methods, data, and efficacy. *Alzheimer's & Dementia: Diagnosis, Assessment & Disease Monitoring*, 17(1), e70082.

Vardhan, K., & Jayashree, R. (2025). Multimodal framework for early detection of Alzheimer's disease using acoustic and linguistic features. In *Intelligent Computing - Proceedings of the Computing Conference* (pp. 588-599).
