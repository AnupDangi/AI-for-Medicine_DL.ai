# Course 2 · Week 3 — Survival Models and Time

> So far we predicted *whether* an event happens. **Survival analysis** predicts *when* — and copes
> with the central challenge of medical follow-up data: **censoring**, where we lose track of patients
> before the event occurs. You'll build the **Kaplan-Meier** survival estimator.

---

## Learning objectives

1. Define the **survival function** `S(t) = P(T > t)`.
2. Understand **right-censoring** and why it can't be ignored.
3. See why a **naïve estimator** that drops or mishandles censored patients is biased.
4. Derive and implement the **Kaplan-Meier** estimator.
5. Compare subgroups and (bonus) test differences with the **log-rank test**.

---

## 1. The survival function

Survival analysis models **time-to-event** data. The core object is the **survival function**:
```
S(t) = P(T > t)   — the probability a patient survives (event-free) beyond time t
```
`S(0) = 1` and it decreases over time. We want to estimate this curve from data.

---

## 2. Censoring — the defining challenge

In a study, many patients **never experience the event during the observation window**:
- The study ends before their event.
- They drop out / move away ("lost to follow-up").

These are **right-censored**: we only know the event happened *after* the last time we saw them, not
exactly when (or if).

**Why you can't just ignore it:**
- Deleting censored patients throws away real information and **biases** the estimate.
- Treating "censored" as "survived forever" **overestimates** survival.
- Treating "censored" as "event happened now" **underestimates** survival.

The lab walks through carefully counting patients into groups: definitely survived past `t`, may have
survived past `t`, and those not censored before `t`.

> Labs: `C2_W3_Lab_1_counting_patients.ipynb` · Exercise: `frac_censored`

---

## 3. Naïve estimator (and why it falls short)

A **naïve estimator** of `S(t)` might be:
```
S(t) ≈ (# known to survive past t) / (total patients)
```
This mishandles censored patients — it either ignores their partial information or misclassifies them,
producing a biased survival curve. This motivates a principled estimator.

> Assignment exercise: `naive_estimator`

---

## 4. The Kaplan-Meier estimator

Kaplan-Meier estimates `S(t)` **without assuming any parametric form**, correctly using censored data.
The idea: survival to time `t` is the product of surviving each event time up to `t`.

At each distinct event time `tᵢ`:
- `dᵢ` = number of events (e.g. deaths) at `tᵢ`
- `nᵢ` = number **at risk** just before `tᵢ` (alive and not yet censored)

```
S(t) = ∏  ( 1 − dᵢ / nᵢ )
      tᵢ ≤ t
```

Censored patients **stay in the at-risk count** until their censoring time, then simply leave the
risk set — so their partial information is used, and the curve only steps down at actual events. This
produces the familiar descending **step function**.

> Labs: `C2_W3_Lab_2_kaplan_meier.ipynb` · Assignment exercise: `HomemadeKM`

---

## 5. Subgroup analysis & the log-rank test

Plot separate Kaplan-Meier curves for subgroups (e.g. treatment vs control, stage I vs stage IV) to
compare survival. The **log-rank test** (bonus) gives a p-value for whether two survival curves differ
significantly — the standard hypothesis test in survival studies.

---

## Assignment

**`C2W3_Assignment.ipynb` — Survival Estimates that Vary with Time**

Compute the fraction censored, build a naïve estimator, implement Kaplan-Meier from scratch
(`HomemadeKM`), then run subgroup analysis and an optional log-rank test.

---

## Key terms cheat-sheet

| Term | Meaning |
|------|---------|
| Time-to-event | Outcome is *when* an event occurs |
| Survival function S(t) | P(patient survives past time t) |
| Right-censoring | Event time only known to be after last contact |
| At-risk set (nᵢ) | Patients still observable just before time tᵢ |
| Kaplan-Meier | Non-parametric survival estimator handling censoring |
| Log-rank test | Test for difference between two survival curves |

---

## Sources & further reading

- [AI for Medical Prognosis — Course home](https://www.coursera.org/learn/ai-for-medical-prognosis)
- [Kaplan-Meier estimator (Wikipedia)](https://en.wikipedia.org/wiki/Kaplan%E2%80%93Meier_estimator)
- [lifelines — survival analysis in Python](https://lifelines.readthedocs.io/en/latest/)
- [Censoring in survival analysis](https://en.wikipedia.org/wiki/Censoring_(statistics))
- [Log-rank test](https://en.wikipedia.org/wiki/Logrank_test)
