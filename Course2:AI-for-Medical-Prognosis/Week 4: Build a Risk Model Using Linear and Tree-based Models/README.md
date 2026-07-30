# Course 2 · Week 4 — Cox Models & Random Survival Forests

> The capstone of prognosis: combine survival analysis with regression. You'll fit the **Cox
> Proportional Hazards** model to get per-patient **hazard ratios**, extend the C-index to censored
> data (**Harrell's C-index**), and use **Random Survival Forests** for non-linear survival prediction.

---

## Learning objectives

1. Encode **categorical variables** with one-hot encoding (avoiding multicollinearity).
2. Define the **hazard function** and the **survival ↔ hazard** relationship.
3. Fit and interpret a **Cox Proportional Hazards** model.
4. Compute and interpret **hazard ratios** between patients.
5. Evaluate censored survival predictions with **Harrell's C-index**.
6. Model non-linear survival with **Random Survival Forests** and interpret via **permutation importance**.

---

## 1. Categorical variables & one-hot encoding

Models need numbers, but features like *race* or *disease stage* are categories with no natural order.
**One-hot encoding** turns a category with `k` levels into binary columns.

- **Drop one column** (`drop_first=True`) to avoid the **dummy variable trap** — perfect
  multicollinearity where the columns sum to 1 and become redundant/unstable for linear models.
- Cast the resulting indicators to floats so the model treats them numerically.

> Labs: `C2_W4_Lab_1_categorical_variables.ipynb` · Assignment exercise: `to_one_hot`

---

## 2. The hazard function

The **hazard** `λ(t)` is the *instantaneous* rate of the event at time `t`, given survival up to `t`:
```
λ(t) = P(event in [t, t+dt] | survived to t) / dt
```
Think "how risky is this exact moment for a patient still alive." Hazard and survival are linked:
```
S(t) = exp( − ∫₀ᵗ λ(u) du )
```
High hazard → survival curve drops steeply.

> Labs: `C2_W4_Lab_2_hazard_function.ipynb`

---

## 3. Cox Proportional Hazards model

Cox regression models how patient features scale the hazard, **without** assuming a shape for the
baseline hazard:
```
λ(t | x) = λ₀(t) · exp(w₁x₁ + w₂x₂ + ... + wₙxₙ)
```
- `λ₀(t)` = **baseline hazard** (shared, unspecified — this is what makes Cox "semi-parametric").
- `exp(w·x)` = how the patient's features **multiply** that baseline risk.

**Proportional hazards assumption:** the ratio of hazards between any two patients is **constant over
time** — their risk curves are proportional, never crossing.

> Assignment sections: **4–5. Cox Proportional Hazards / Fitting and Interpreting**

---

## 4. Hazard ratio

Comparing two patients A and B, the baseline `λ₀(t)` cancels:
```
HR = λ(t | x_A) / λ(t | x_B) = exp( w · (x_A − x_B) )
```
- **HR > 1**: patient A has higher risk. **HR = 2** → twice the instantaneous risk.
- **HR < 1**: lower risk. **HR = 1**: same risk.
- For a single feature, `exp(wᵢ)` is the hazard ratio per one-unit increase — this is how Cox
  coefficients are interpreted clinically.

> Labs: `C2_W4_Lab_2_hazard_function.ipynb` · Assignment exercise: `hazard_ratio`

---

## 5. Harrell's C-index (concordance with censoring)

Week 1's C-index assumed we knew every outcome. With **censoring**, not every pair is comparable.
**Harrell's C-index** only counts **permissible pairs** — pairs where we can actually tell who had the
event first given the censoring times.

- A pair is **permissible** if the earlier time is an actual event (not censored).
- **Concordant** if the patient with the shorter survival has the higher risk score; ties count half.
```
Harrell's C = (concordant + 0.5 · ties) / permissible pairs
```
Same 0.5-to-1.0 interpretation, but valid for time-to-event data.

> Labs: `C2_W4_Lab_3_permissible_pairs_with_censoring_and_time.ipynb` · Assignment exercise: `harrell_c`

---

## 6. Random Survival Forests & permutation importance

The Cox model assumes linear, proportional effects. **Random Survival Forests (RSF)** extend random
forests to censored survival data, capturing **non-linearities and interactions** automatically —
often improving the C-index over Cox.

Because RSF is a black box, we interpret it with **permutation importance**: shuffle one feature's
values and measure how much the C-index drops. A big drop ⇒ that feature was important.

> Assignment sections: **8–9. Random Survival Forests / Permutation Method**

---

## Assignment

**`C2W4_Assignment.ipynb` — Cox Proportional Hazards and Random Survival Forests**

Dataset: primary biliary cirrhosis (`pbc`). You'll one-hot encode features, fit and interpret a Cox
model, compute hazard ratios, implement Harrell's C-index, then fit a Random Survival Forest and rank
features by permutation importance.

---

## Key terms cheat-sheet

| Term | Meaning |
|------|---------|
| One-hot encoding | Binary columns for categorical levels |
| Dummy variable trap | Multicollinearity from keeping all one-hot columns |
| Hazard λ(t) | Instantaneous event rate given survival to t |
| Cox PH model | λ₀(t)·exp(w·x); semi-parametric survival regression |
| Proportional hazards | Hazard ratio between patients is constant over time |
| Hazard ratio | exp(w·Δx); relative instantaneous risk |
| Harrell's C-index | C-index restricted to permissible (censoring-aware) pairs |
| Random Survival Forest | Tree ensemble for censored survival data |
| Permutation importance | Feature importance via shuffling and measuring performance drop |

---

## Sources & further reading

- [AI for Medical Prognosis — Course home](https://www.coursera.org/learn/ai-for-medical-prognosis)
- [Cox Proportional Hazards (lifelines)](https://lifelines.readthedocs.io/en/latest/Survival%20Regression.html)
- [Cox model (Wikipedia)](https://en.wikipedia.org/wiki/Proportional_hazards_model)
- [Random Survival Forests (Ishwaran et al., 2008)](https://arxiv.org/abs/0811.1645)
- [scikit-survival library](https://scikit-survival.readthedocs.io/en/stable/)
- [Harrell's concordance index](https://en.wikipedia.org/wiki/Concordance_correlation_coefficient)
