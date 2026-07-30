# Course 1 · Week 2 — Evaluating Models

> A model that "works" is not the same as a model that is *safe to deploy in a clinic*.
> This week is a deep dive into the metrics that tell you whether a diagnostic model can be trusted:
> sensitivity, specificity, PPV/NPV, ROC/AUC, precision-recall, F1, calibration, and confidence intervals.

---

## Learning objectives

1. Build the **confusion matrix** (TP, FP, TN, FN) and reason about it in clinical terms.
2. Compute and interpret **accuracy, prevalence, sensitivity, specificity, PPV, NPV**.
3. Understand how these metrics relate via **Bayes' rule / conditional probability**.
4. Use the **ROC curve** and **threshold (operating point)** to trade off errors.
5. Read a **precision-recall curve**, compute **F1**, and check **calibration**.
6. Interpret **95% confidence intervals** correctly.

---

## 1. The confusion matrix

Every prediction on a binary task falls into one of four buckets:

|  | Actually Positive | Actually Negative |
|---|---|---|
| **Predicted Positive** | True Positive (TP) | False Positive (FP) |
| **Predicted Negative** | False Negative (FN) | True Negative (TN) |

In medicine: a **False Negative** (missing a real disease) is often far more dangerous than a
**False Positive** (a false alarm) — which is why accuracy alone is a poor metric.

---

## 2. Core metrics

### Accuracy
```
Accuracy = (TP + TN) / (TP + TN + FP + FN)
```
Misleading under class imbalance (see Week 1). It can be decomposed as:
```
Accuracy = Sensitivity · Prevalence + Specificity · (1 − Prevalence)
```

### Prevalence
```
Prevalence = (TP + FN) / total = fraction of patients who actually have the disease
```

### Sensitivity (Recall / True Positive Rate) — "of the sick, how many did we catch?"
```
Sensitivity = TP / (TP + FN)
```

### Specificity (True Negative Rate) — "of the healthy, how many did we clear?"
```
Specificity = TN / (TN + FP)
```

Sensitivity and specificity are **properties of the model** and do not depend on prevalence.

### PPV & NPV — the questions a patient actually cares about
- **Positive Predictive Value**: "I tested positive — what's the chance I'm actually sick?"
  ```
  PPV = TP / (TP + FP)
  ```
- **Negative Predictive Value**: "I tested negative — what's the chance I'm actually healthy?"
  ```
  NPV = TN / (TN + FN)
  ```
- **PPV and NPV depend heavily on prevalence.** Using Bayes' rule:
  ```
  PPV = (Sens · Prev) / (Sens · Prev + (1 − Spec) · (1 − Prev))
  ```

---

## 3. ROC curve, threshold, and AUC

A classifier outputs a **probability**. To turn it into a yes/no decision you pick a **threshold**
(the *operating point*). Moving the threshold trades sensitivity against specificity:

- Lower threshold → catch more disease (↑ sensitivity) but more false alarms (↓ specificity).
- Higher threshold → fewer false alarms (↑ specificity) but miss more cases (↓ sensitivity).

The **ROC curve** plots **TPR (sensitivity)** vs **FPR (1 − specificity)** across all thresholds.
The **AUC (Area Under the Curve)** summarizes it in one number:
- 1.0 = perfect ranking, 0.5 = random guessing.
- Interpretation: the probability the model ranks a random positive above a random negative.

> Lab: `C1_W2_Lab_1_roc_curve_and_threshold.ipynb`

---

## 4. Precision-Recall, F1, and Calibration

- **Precision-Recall curve**: precision (= PPV) vs recall (= sensitivity). More informative than ROC
  when the positive class is rare.
- **F1 score**: harmonic mean of precision and recall — one number balancing the two:
  ```
  F1 = 2 · (Precision · Recall) / (Precision + Recall)
  ```
- **Calibration**: are the predicted probabilities *honest*? If the model says "0.7", do ~70% of such
  cases truly have the disease? A **calibration plot** compares predicted vs observed frequencies.

---

## 5. Confidence intervals

Metrics are estimated from a *sample*, so they carry uncertainty. A **95% confidence interval** means:
if we repeated the study many times, ~95% of the computed intervals would contain the true population
value. Key intuition: **larger test sets → narrower intervals** (more certainty). A CI does *not* say
"there's a 95% chance the true value is in this specific interval."

---

## Assignment

**`C1W2_Assignment.ipynb` — Evaluation of Diagnostic Models**

Implement, from scratch: TP/FP/TN/FN counting, accuracy, prevalence, sensitivity, specificity, PPV,
NPV, then plot ROC curves, precision-recall curves, compute F1, and assess calibration.

---

## Key terms cheat-sheet

| Term | Formula / meaning |
|------|-------------------|
| Sensitivity (Recall/TPR) | TP / (TP + FN) |
| Specificity (TNR) | TN / (TN + FP) |
| Precision (PPV) | TP / (TP + FP) |
| NPV | TN / (TN + FN) |
| Prevalence | positives / total |
| ROC / AUC | TPR vs FPR; area = ranking quality |
| F1 | harmonic mean of precision & recall |
| Calibration | predicted probability ≈ observed frequency |
| Confidence interval | plausible range for a metric given sampling |

---

## Sources & further reading

- [AI for Medical Diagnosis — Course home](https://www.coursera.org/learn/ai-for-medical-diagnosis)
- [scikit-learn: ROC & AUC](https://scikit-learn.org/stable/modules/model_evaluation.html#roc-metrics)
- [scikit-learn: Precision-Recall](https://scikit-learn.org/stable/auto_examples/model_selection/plot_precision_recall.html)
- [Sensitivity & specificity (Wikipedia)](https://en.wikipedia.org/wiki/Sensitivity_and_specificity)
- [Calibration of probabilities (scikit-learn)](https://scikit-learn.org/stable/modules/calibration.html)
