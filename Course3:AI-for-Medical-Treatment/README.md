# Course 3 — AI for Medical Treatment

> The final course: from *diagnosis* (Course 1) and *prognosis* (Course 2) to **treatment** — using
> data to decide *which patients actually benefit from an intervention*, extract structured facts from
> free-text reports with NLP, and explain model decisions.
>
> **Status in this repo:** **Week 1 is complete** (3 labs + graded assignment) — see the dedicated
> [Week 1 README](./Week%201%3A%20Treatment%20Effect%20Estimation/README.md). Weeks 2–3 are taught below
> from the official syllabus so the concepts are ready to reference as those notebooks are added.

---

## Week 1 — Treatment Effect Estimation ✅

> Full write-up with formulas, lab-by-lab breakdown, and sources:
> **[Week 1: Treatment Effect Estimation → README](./Week%201%3A%20Treatment%20Effect%20Estimation/README.md)**

**Big idea:** a prognostic model tells you a patient's *risk*; a **treatment effect** model tells you how
much a *treatment changes* that risk — the quantity a doctor needs to decide whether to treat.

- **RCTs** randomly assign patients to treatment/control arms, enabling **causal** effect estimates.
- **Odds ratio (OR)** quantifies a constant treatment effect; convert it to **Absolute Risk Reduction
  (ARR = risk(control) − risk(treated))**, which varies with baseline risk (`NNT = 1/ARR`).
- **C-statistic-for-benefit** evaluates how well a model ranks *who benefits most*.
- **T-Learner** estimates individualized benefit: train `μ₁` (treated) and `μ₀` (control); benefit = `μ₀(x) − μ₁(x)`.

**Notebooks:** `Lab 1` pandas for medical data · `Lab 2` sklearn training/tuning & grid search ·
`Lab 3` logistic regression interpretation (odds, logit, odds ratio) · **Assignment** *Estimating
Treatment Effect Using Machine Learning*.

---

## Week 2 — Medical Question Answering & NLP

### Big idea
Most clinical information is locked in **unstructured text** (radiology reports, notes). This week
automates extracting structured labels and answering questions from that text.

### Automatic label extraction from reports
- **Rules-based / text-matching** extraction: search reports for mentions of conditions.
- **Negation detection:** distinguish "*no evidence of pneumonia*" (negative) from "*pneumonia present*"
  (positive) — critical to avoid false labels.
- **Dependency parsing:** analyze grammatical structure so negations/uncertainty attach to the right
  finding (tools like the **NegBio** library, built on the BioC format).
- **Evaluating automatic labeling:** compare extracted labels to gold-standard annotations
  (precision/recall/F1).

### Question Answering with BERT
- **BERT** (Bidirectional Encoder Representations from Transformers) is a pre-trained language model.
- For **extractive QA**, BERT reads a question + a passage and predicts the **start and end tokens** of
  the answer span within the passage.
- Applied to medicine: ask "*What is the patient's blood pressure?*" and pull the answer from the note.

**Assignment:** *Natural Language Entity Extraction* — build a label extractor with negation handling
and evaluate it.

---

## Week 3 — Machine Learning Interpretation

### Big idea
For clinical adoption, models must be **explainable**. This week covers how to interpret both tabular
models and deep vision models.

### Feature importance
- **Drop-column / individual feature importance:** measure performance change when a feature is removed.
- **Permutation importance:** shuffle a feature's values and measure the drop in performance — a
  model-agnostic global importance measure.
- **Shapley (SHAP) values:** game-theoretic per-prediction attributions that fairly distribute the
  prediction among features and sum exactly to it (local + global explanations).

### Interpreting deep learning models
- **Grad-CAM (Gradient-weighted Class Activation Mapping):** produce a heatmap over a medical image
  showing which regions drove the prediction — connecting back to the chest-X-ray model of Course 1.

**Assignment:** *ML Interpretation* — apply permutation importance, SHAP, and Grad-CAM to explain
diagnostic/prognostic models.

---

## Key terms cheat-sheet

| Term | Meaning |
|------|---------|
| RCT | Randomized control trial; gold standard for causal effect |
| ARR | Absolute risk reduction = risk(control) − risk(treated) |
| ATE / ITE | Average vs individualized treatment effect |
| NNT | Number needed to treat = 1/ARR |
| T-Learner / S-Learner | Two-model / single-model estimators of treatment effect |
| C-for-benefit | Concordance metric for treatment benefit |
| Negation detection | Telling "no pneumonia" from "pneumonia" in text |
| BERT | Transformer LM used here for extractive question answering |
| Permutation importance | Feature importance via shuffling |
| SHAP | Shapley-value feature attributions |
| Grad-CAM | Heatmap explaining image-model predictions |

---

## Sources & further reading

- [AI for Medical Treatment — Course home](https://www.coursera.org/learn/ai-for-medical-treatment)
- [Specialization outline](https://www.deeplearning.ai/specializations/ai-for-medicine)
- [BERT paper (Devlin et al., 2018)](https://arxiv.org/abs/1810.04805)
- [NegBio (negation & uncertainty detection)](https://github.com/ncbi-nlp/NegBio)
- [SHAP — Lundberg & Lee, 2017](https://arxiv.org/abs/1705.07874)
- [Grad-CAM — Selvaraju et al., 2017](https://arxiv.org/abs/1610.02391)
- [Random Forests & permutation importance (Breiman, 2001)](https://www.stat.berkeley.edu/~breiman/randomforest2001.pdf)
- [T-Learner / meta-learners for CATE (Künzel et al., 2019)](https://arxiv.org/abs/1706.03461)
- [pandas documentation](https://pandas.pydata.org/docs/)
