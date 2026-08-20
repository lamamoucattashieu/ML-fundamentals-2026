# Bank Marketing — Classification on an Imbalanced Target

A supervised ML pipeline on the UCI Bank Marketing dataset, predicting which
clients subscribe to a term deposit. The target is heavily imbalanced
(~11% positive), which makes this a case study in a lesson that matters more
than any single model: **why accuracy is the wrong metric here, and what to use
instead — with leakage-safe preprocessing at every step.**

## The core finding

A zero-rule baseline that *never* predicts a subscriber scores **0.89 accuracy**
by exploiting the class imbalance alone. A logistic regression scores a lower
**0.78 accuracy**, yet recovers **~60% of actual subscribers** (recall 0.60,
F1 0.37).

On accuracy alone the useless baseline "wins." That inversion is the point: for
an imbalanced business target, **recall and F1 decide which model is worth
deploying**, not accuracy.

## What the notebook covers

Structured as a full, leakage-aware pipeline with the reasoning made explicit
at each step:

- **Target selection** — chooses the prediction target and argues why `duration`
  and `poutcome` are excluded (leakage / post-outcome information).
- **EDA** — missing-value analysis, skewness and distributions, the `pdays` 999
  sentinel handled explicitly.
- **Leakage-safe splitting** — stratified split *before* any fitting, with a
  written account of every leakage path that a later split would open.
- **Preprocessing** — categorical encoding, feature scaling fit on the training
  set only, low-variance and high-correlation feature removal (threshold 0.8),
  each justified as training-set-only.
- **Class imbalance** — random oversampling on the training set only, with a
  discussion of why resampling before splitting corrupts evaluation.
- **Model + evaluation** — logistic regression vs. a zero-rule baseline, scored
  on accuracy / precision / recall / F1 to expose the accuracy trap above.
- **Pipeline ordering** — a final section on correct task order and how
  reordering introduces leakage.

## Tech stack

Python · scikit-learn · imbalanced-learn · pandas · NumPy · Matplotlib

## Notes

Built for the Machine Learning Fundamentals course at IE University. The emphasis
is evaluation discipline and leakage-safe methodology on imbalanced data, not
model complexity — the "right" model is the one that finds subscribers, not the
one with the highest accuracy.

---

*Full analysis in `assignment1_lama_moucattash.ipynb`.*
