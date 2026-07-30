# Course 2 · Week 1 — Linear Prognostic Models

> **Prognosis** = predicting the *future* course of a patient's health (risk of an event) rather than
> diagnosing a current disease. This week builds your first **risk model**: a logistic-regression
> model that outputs a risk score, plus the tools to engineer features and evaluate it with the
> **C-index**.

---

## Learning objectives

1. Understand what a **prognostic model / risk score** is and where they're used clinically.
2. Build a **linear model** and a **logistic regression** risk model with scikit-learn.
3. **Mean-normalize** (standardize) features correctly using training statistics.
4. Engineer **feature interactions** and know why we *multiply* features rather than add them.
5. Evaluate a risk model with the **Concordance Index (C-index)**.

---

## 1. What is a prognostic model?

A prognostic model takes patient features (age, blood pressure, lab values, ...) and outputs a
**risk score** — a number where higher = more likely to experience an adverse event (death, stroke,
disease onset) within some time frame.

Real-world examples you'll meet this week:
- **Atrial fibrillation / stroke risk**, **liver disease (MELD-style) scores**, and the
  **ASCVD score** for 10-year cardiovascular risk.

These classic clinical scores are essentially **linear combinations of features** passed through a
function — exactly what logistic regression learns.

> Labs: `C2_W1_Lab_2_risk_scores_pandas_and_numpy.ipynb`

---

## 2. Linear & logistic risk models

### Linear model
```
score = w₁x₁ + w₂x₂ + ... + wₙxₙ + b
```
Simple and interpretable — each weight `wᵢ` tells you how feature `i` pushes risk up or down.

### Logistic regression (the risk model)
A raw linear score can be any real number. To turn it into a **probability in [0, 1]** we pass it
through the **sigmoid**:
```
p = σ(w·x + b) = 1 / (1 + e^{-(w·x + b)})
```
`p` is the estimated probability of the adverse event. This is the core prognostic model of the week.

> Labs: `C2_W1_Lab_1_create_a_linear_model.ipynb`

---

## 3. Mean-normalization (standardization)

Features live on wildly different scales (age ~ 60, some lab value ~ 0.001). Linear models train
better when every feature is centered and scaled:
```
x_scaled = (x − μ_train) / σ_train
```
**Critical rule:** compute `μ` and `σ` on the **training set only**, then apply those *same* values to
the test set. Recomputing statistics on the test data leaks information and inflates performance. In the
assignment you also log-transform skewed features first to make them more Gaussian.

> Assignment exercise: `make_standard_normal`

---

## 4. Feature interactions

A plain linear model assumes each feature acts independently. But risk factors often **combine** —
e.g. high blood pressure is more dangerous *at older ages*. We capture this by adding **interaction
terms**:
```
x_interaction = xᵢ × xⱼ
```
- We **multiply** (not add) because multiplication encodes "the effect of one feature *depends on* the
  value of another." Adding two features just creates another linear term that the model could already
  represent.
- Adding interactions lets a linear model capture some **non-linear** relationships, usually improving
  the C-index.

> Labs: `C2_W1_Lab_3_combine_features.ipynb` · Assignment exercise: `add_interactions`

---

## 5. Evaluating with the Concordance Index (C-index)

For risk models we care about **ranking**: does the model give sicker patients higher risk scores?
The **C-index** measures exactly that.

- Look at every **permissible pair** of patients (one had the event, one did not).
- A pair is **concordant** if the patient who had the event received the higher risk score.
- **Ties** (equal risk scores) count as half.
```
C-index = (concordant + 0.5 · ties) / permissible pairs
```
Interpretation is like AUC: **0.5 = random**, **1.0 = perfect** ranking. In fact, for binary outcomes
the C-index equals the AUROC.

> Labs: `C2_W1_Lab_4_concordance_index.ipynb` · Assignment exercise: `cindex`

---

## Assignment

**`C2W1_Assignment.ipynb` — Build and Evaluate a Linear Risk Model**

Dataset: **diabetic retinopathy**. You'll standardize features, fit a logistic-regression risk model,
evaluate with the C-index on a held-out test set, then improve it by adding feature interactions and
re-measuring.

---

## Key terms cheat-sheet

| Term | Meaning |
|------|---------|
| Prognosis | Predicting the future course / risk of an event |
| Risk score | Model output; higher = more likely adverse event |
| Logistic regression | Linear model + sigmoid → probability |
| Standardization | (x − μ)/σ using **training** statistics |
| Interaction term | Product xᵢ·xⱼ capturing combined effects |
| Concordant pair | Higher-risk patient is the one who had the event |
| C-index | Ranking quality; 0.5 random, 1.0 perfect |

---

## Sources & further reading

- [AI for Medical Prognosis — Course home](https://www.coursera.org/learn/ai-for-medical-prognosis)
- [scikit-learn LogisticRegression](https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LogisticRegression.html)
- [Concordance index (lifelines docs)](https://lifelines.readthedocs.io/en/latest/lifelines.utils.html#lifelines.utils.concordance_index)
- [ASCVD risk estimator (ACC/AHA)](https://tools.acc.org/ascvd-risk-estimator-plus/)
- [Feature scaling (scikit-learn)](https://scikit-learn.org/stable/modules/preprocessing.html)
