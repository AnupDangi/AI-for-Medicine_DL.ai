# Course 3 · Week 1 — Treatment Effect Estimation

> Diagnosis (Course 1) asks *"what does this patient have?"* and prognosis (Course 2) asks *"what will
> happen to them?"*. **Treatment effect estimation** asks the question a clinician actually acts on:
> *"how much will this treatment help **this** patient?"* You'll analyze **randomized control trial
> (RCT)** data with both classical statistics and machine learning, and learn to estimate benefit at
> the **individual** level.

---

## Learning objectives

By the end of this week you should be able to:

1. Explain why **RCTs** enable *causal* conclusions and process RCT data (`proportion_treated`, `event_rate`).
2. Quantify a **constant treatment effect** using the **odds ratio (OR)**.
3. Convert an odds ratio into an **absolute risk reduction (ARR)** and see why benefit depends on baseline risk.
4. Understand the limits of a constant-effect model and estimate **baseline risk** per patient.
5. Evaluate treatment-benefit models with the **C-statistic-for-benefit (C-for-benefit)**.
6. Estimate **individualized treatment effects** with a **T-Learner**, tuned via **grid search**.

---

## 0. Foundations from the labs

Before the modeling, three lecture notebooks build the toolkit:

- **Lab 1 — `C3_W1_Lab_1_pandas_for_a_medical_dataset.ipynb`**
  pandas essentials for tabular clinical data: `DataFrame` vs `Series`, slicing, boolean **masks**
  (combine conditions with `&` / `|` and **parentheses** — plain `and` fails on Series), `.mean()`, and
  safely updating values with `.loc`.

- **Lab 2 — `C3_W1_Lab_2_model_training_basics_with_sklearn.ipynb`**
  the scikit-learn workflow: **train/test split**, `.fit()`, `.predict()` vs `.predict_proba()`, mean
  accuracy, and **grid search** for hyperparameter tuning — exactly the machinery used to build the
  treatment-effect models later.

- **Lab 3 — `C3_W1_Lab_3_logistic_regression_model_interpretation (1).ipynb`**
  how to *read* a logistic regression: **odds**, the **logit** (log-odds), how a coefficient shifts the
  logit, and turning coefficients into **odds ratios**. This is the statistical backbone of the assignment.

---

## 1. Why randomized control trials (RCTs)?

In an RCT, patients are **randomly assigned** to a **treatment** arm or a **control** arm. Randomization
makes the two groups statistically similar in every respect *except* the treatment, so any difference in
outcomes can be attributed **causally** to the treatment — not to confounding factors. This is why RCTs
are the gold standard for measuring "does the treatment work?"

Data processing this week:
- **`proportion_treated`** — fraction of patients in the treatment arm.
- **`event_rate`** — fraction who experienced the adverse outcome (used per-arm to compare risk).

---

## 2. Quantifying a constant treatment effect: the odds ratio

### Odds, logit, and odds ratio (from Lab 3)
- **Probability** `p` → **odds** `= p / (1 − p)`.
- **Logit** `= log(odds)` — logistic regression is linear in the logit.
- An **odds ratio (OR)** compares the odds of an event between two conditions:
```
OR = odds(treated) / odds(control)
```
- **OR < 1** → treatment *reduces* the odds of the (bad) event (beneficial).
- **OR = 1** → no effect. **OR > 1** → treatment increases the event odds.

In logistic regression, adding a **treatment indicator** feature and taking `exp(coefficient)` gives the
treatment odds ratio directly (`extract_treatment_effect`).

---

## 3. From odds ratio to Absolute Risk Reduction (ARR)

An odds ratio alone doesn't tell a patient how much they benefit — that depends on their **baseline
risk**. **Absolute Risk Reduction** is the intuitive quantity:
```
ARR = risk(control) − risk(treated)
```
Given a baseline risk `p` and an odds ratio, you can convert (`OR_to_ARR`):
```
baseline odds = p / (1 − p)
treated odds  = OR · baseline odds
treated risk  = treated odds / (1 + treated odds)
ARR           = p − treated risk
```

**Key insight (visualized in the assignment):** with a *constant* odds ratio, the ARR is **not
constant** — it varies with baseline risk. Patients at moderate baseline risk often gain the most
absolute benefit, while very low-risk patients gain little. `Number Needed to Treat = 1 / ARR`.

---

## 4. Model limitations & baseline risk

A single constant-OR model assumes every patient responds identically — usually false. The assignment
has you:
- Estimate each patient's **baseline risk** (`base_risks`) from a logistic-regression risk model.
- Bin patients by risk and compare **ARR across quantiles** (`lr_ARR_quantile`) to reveal that benefit
  is heterogeneous across the population. This motivates *individualized* modeling.

---

## 5. Evaluating benefit: C-statistic-for-benefit

The ordinary C-index measures how well you rank patients by **risk**. But we want to rank patients by
**benefit** — and that's harder because we **never observe both outcomes** for the same patient (they
were either treated *or* control, not both — the fundamental problem of causal inference).

**C-for-benefit** solves this by **pairing** a treated patient with a similar untreated patient to form a
"virtual" observation of benefit, then measuring concordance:
- A pair is **concordant** if the pair with the larger *predicted* benefit also had the larger *observed*
  benefit; ties count half.
```
C-for-benefit = (concordant + 0.5 · ties) / permissible pairs
```
- **0.5 = no better than random**, **1.0 = perfectly ranks who benefits most**.

Assignment exercises: `c_for_benefit_score`, `c_statistic` (plus comparison against the regular c-index).

---

## 6. Individualized treatment effect: the T-Learner

The **T-Learner** ("two-model" method) estimates the **Conditional Average Treatment Effect (CATE)** for
each patient:

1. Train model **μ₁** on the **treated** arm only → predicts risk if treated.
2. Train model **μ₀** on the **control** arm only → predicts risk if untreated.
3. For any patient `x`:
```
estimated benefit(x) = μ₀(x) − μ₁(x)     # reduction in risk from treatment
```
Base learners can be any model (here, tree-based/`RandomForest`-style estimators). The models are tuned
with **grid search** over a held-out set:
- `holdout_grid_search` — search hyperparameters and keep the best by validation score.
- `treatment_dataset_split` — split data into treated/control subsets to train μ₁ and μ₀.

The result: instead of one population-wide number, you get a **personalized** benefit estimate per patient.

---

## Assignment

**`C3W1_Assignment.ipynb` — Estimating Treatment Effect Using Machine Learning**

End-to-end: process RCT data, compute the treatment odds ratio, convert OR→ARR, analyze how benefit
varies with baseline risk, implement the C-for-benefit metric, then build and grid-search a T-Learner to
estimate individualized treatment effects.

---

## Key terms cheat-sheet

| Term | Meaning |
|------|---------|
| RCT | Randomized control trial; enables causal effect estimates |
| Event rate | Fraction of an arm that had the adverse outcome |
| Odds / logit | p/(1−p) and its log; logistic regression is linear in logit |
| Odds ratio (OR) | odds(treated) / odds(control); OR<1 = benefit |
| Absolute Risk Reduction (ARR) | risk(control) − risk(treated) |
| Number Needed to Treat | 1 / ARR |
| Baseline risk | A patient's risk without treatment |
| C-for-benefit | Concordance for ranking treatment *benefit* |
| T-Learner | Two models (μ₁ treated, μ₀ control); benefit = μ₀−μ₁ |
| CATE | Conditional Average Treatment Effect (per-patient benefit) |
| Grid search | Exhaustive hyperparameter tuning on a holdout set |

---

## Citations (Week 1)

- [Levamisole and Fluorouracil background](https://www.nejm.org/doi/full/10.1056/NEJM199002083220602) — Moertel et al., *N Engl J Med*, 1990
- [Data sourced from here](https://www.rdocumentation.org/packages/survival/versions/3.1-8/topics/colon) — `survival::colon` RCT dataset (R)
- [C-statistic for benefit](https://www.ncbi.nlm.nih.gov/pubmed/29132832) — van Klaveren et al., *J Clin Epidemiol*, 2018
- [T-learner](https://arxiv.org/pdf/1706.03461.pdf) — Künzel et al., meta-learners for CATE, 2019

### Additional references

- [AI for Medical Treatment — Course home](https://www.coursera.org/learn/ai-for-medical-treatment)
- [Randomized controlled trial (Wikipedia)](https://en.wikipedia.org/wiki/Randomized_controlled_trial)
- [Odds ratio (Wikipedia)](https://en.wikipedia.org/wiki/Odds_ratio)
- [Absolute vs relative risk reduction](https://en.wikipedia.org/wiki/Absolute_risk_reduction)
- [scikit-learn: LogisticRegression](https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LogisticRegression.html)
- [scikit-learn: GridSearchCV](https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.GridSearchCV.html)
- [pandas documentation](https://pandas.pydata.org/docs/)
