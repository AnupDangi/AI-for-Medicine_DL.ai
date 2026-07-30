# Course 2 · Week 2 — Prognosis with Tree-based Models

> Linear models are simple but limited. This week you graduate to **decision trees** and **random
> forests**, learn to handle the **missing data** that plagues real medical records, and explain the
> resulting "black-box" model with **SHAP** values.

---

## Learning objectives

1. Build and tune a **decision tree** classifier for risk prediction.
2. Understand **overfitting** in trees and how hyperparameters control it.
3. Combine trees into a **random forest** and tune it with grid search.
4. Diagnose and handle **missing data** (MCAR / MAR / MNAR) and apply **masks**.
5. Fill gaps with **mean imputation** and **regression (IterativeImputer) imputation**.
6. Interpret model predictions using **SHAP** values.

---

## 1. Decision trees

A decision tree splits patients by asking a sequence of yes/no questions on features
(e.g. `age > 60?`, `BP < 140?`) until it reaches a leaf that outputs a risk.

- **Pros:** capture **non-linear** relationships and **interactions** automatically; interpretable.
- **Con:** a deep tree **overfits** — it memorizes the training data and generalizes poorly.

### Key hyperparameters (the overfitting dials)
- `max_depth` — how many questions deep the tree can go.
- `min_samples_split` / `min_samples_leaf` — minimum patients required to split / to form a leaf.

Shallower trees + larger leaf sizes = more **regularization** = less overfitting.

> Labs: `C2_W2_Lab_1_decision_tree_classifier.ipynb`

---

## 2. Random forests

A **random forest** trains many decision trees, each on a bootstrap sample of patients and a random
subset of features, then **averages** their predictions.

- **Bagging + feature randomness** decorrelate the trees, so averaging reduces variance dramatically.
- Almost always beats a single tree, at the cost of interpretability.
- Tuned via **grid search** over `n_estimators`, `max_depth`, `min_samples_leaf`, etc.

> Assignment exercises: `decision_tree_classifier`, `random_forest_grid_search`

---

## 3. Missing data — the reality of medical records

Patients don't get every test, so values are missing. **Why** they're missing matters:

- **MCAR (Missing Completely At Random):** missingness unrelated to anything. Dropping rows is safe.
- **MAR (Missing At Random):** missingness depends on *observed* variables (e.g. doctors skip a test
  for younger patients). Droppng rows can **bias** the model.
- **MNAR (Missing Not At Random):** missingness depends on the *unobserved* value itself.

> **Error analysis insight from the assignment:** if you simply drop rows with missing values, the
> resulting subset can be *systematically different* (e.g. skewed toward younger patients), so your
> model's test performance looks fine but is actually biased. This is why imputation matters.

### Masks
A **mask** is a boolean array marking where data is missing (`df.isnull()`), letting you count,
filter, or selectively operate on missing entries.

> Labs: `C2_W2_Lab_2_missing_data_and_applying_a_mask.ipynb` · Exercise: `fraction_rows_missing`, `bad_subset`

---

## 4. Imputation — filling the gaps

- **Mean imputation:** replace missing values with the column mean. Fast, but distorts variance and
  ignores relationships between features.
- **Regression imputation** (scikit-learn `IterativeImputer`): predict each missing feature *from the
  other features* using a regression model. More accurate because it respects correlations.

Always fit the imputer on the **training set** and apply it to the test set (same rule as scaling).

> Labs: `C2_W2_Lab_3_imputation.ipynb`

---

## 5. Explaining the model with SHAP

Random forests are hard to interpret. **SHAP (SHapley Additive exPlanations)** assigns each feature a
contribution to a specific prediction, grounded in cooperative game theory (Shapley values):

- Guarantees the contributions **sum to the prediction** minus the baseline.
- Shows both **global** importance (which features matter overall) and **local** explanations (why
  *this* patient got *this* risk) — vital for clinical trust.

> Assignment section: **7. Explanations: SHAP**

---

## Assignment

**`C2W2_Assignment.ipynb` — Risk Models Using Tree-based Models**

Predict 10-year mortality (NHANES-style data). You'll quantify missingness, train a decision tree and a
random forest with grid search, perform error analysis on dropped-row bias, apply imputation, then
explain predictions with SHAP.

---

## Key terms cheat-sheet

| Term | Meaning |
|------|---------|
| Decision tree | Sequence of feature splits ending in a risk leaf |
| Overfitting | Memorizing training data; poor generalization |
| Random forest | Ensemble of decorrelated trees, averaged |
| Grid search | Exhaustive hyperparameter search |
| MCAR/MAR/MNAR | Categories of why data is missing |
| Mask | Boolean array marking missing entries |
| Imputation | Filling missing values (mean / regression) |
| SHAP | Feature-contribution explanations (Shapley values) |

---

## Sources & further reading

- [AI for Medical Prognosis — Course home](https://www.coursera.org/learn/ai-for-medical-prognosis)
- [Random Forests (Breiman, 2001)](https://www.stat.berkeley.edu/~breiman/randomforest2001.pdf)
- [scikit-learn: Decision Trees](https://scikit-learn.org/stable/modules/tree.html)
- [scikit-learn: IterativeImputer](https://scikit-learn.org/stable/modules/impute.html#iterative-imputer)
- [SHAP documentation](https://shap.readthedocs.io/en/latest/)
- [A Unified Approach to Interpreting Model Predictions (Lundberg & Lee, 2017)](https://arxiv.org/abs/1705.07874)
