# Course 3 — AI for Medical Treatment ✅

> The final course: from *diagnosis* (Course 1) and *prognosis* (Course 2) to **treatment** — using
> data to decide *which patients actually benefit from an intervention*, extract structured facts from
> free-text reports with NLP, and explain model decisions.
>
> **Status in this repo:** **Complete** — Weeks 1–3 (labs + assignments) with dedicated week READMEs.

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

## Week 2 — Medical Question Answering & NLP ✅

> Full write-up:
> **[Week 2: Medical Question Answering → README](./Week%202%3A%20Medical%20Question%20Answering/README.md)**

**Big idea:** most clinical information is locked in **unstructured text**. Automate extracting
structured labels and answering questions from that text.

- **Rules-based / text-matching** extraction, then **negation-aware** rules.
- **BioC + NegBio** (dependency parsing) for negation / uncertainty — toward CheXpert-style labelers.
- Evaluate with **precision / recall / F1**.
- **BERT** extractive QA: question + passage → start/end token span of the answer.

**Notebooks:** `Lab 1` cleaning text (regex) · `Lab 2` BioC & NegBio · `Lab 3` prep input for text
classification · **Ungraded assignment** *Natural Language Entity Extraction* + BERT QA.

---

## Week 3 — Machine Learning Interpretation ✅

> Full write-up:
> **[Week 3: ML Interpretation → README](./Week%203%3A%20ML%20Interpretation/README.md)**

**Big idea:** for clinical adoption, models must be **explainable** — both tabular and vision models.

- **Permutation importance:** shuffle a feature and measure the drop in performance.
- **Shapley (SHAP) values:** per-prediction attributions that sum to the output (local + global).
- **Grad-CAM:** heatmap over a medical image showing which regions drove the CNN prediction.

**Notebooks:** `Lab 1` permutation method · `Lab 2` Grad-CAM intro (activations) · `Lab 3` Grad-CAM
continuation (gradients) · **Assignment** *ML Interpretation* (Grad-CAM + permutation + SHAP).

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

## Citations

Official course reading list (DeepLearning.AI / Coursera), organized by week.

### Week 1 — Treatment Effect Estimation

- [Levamisole and Fluorouracil background](https://www.nejm.org/doi/full/10.1056/NEJM199002083220602) — Moertel et al., *N Engl J Med*, 1990
- [Data sourced from here](https://www.rdocumentation.org/packages/survival/versions/3.1-8/topics/colon) — `survival::colon` RCT dataset (R)
- [C-statistic for benefit](https://www.ncbi.nlm.nih.gov/pubmed/29132832) — van Klaveren et al., *J Clin Epidemiol*, 2018
- [T-learner](https://arxiv.org/pdf/1706.03461.pdf) — Künzel et al., meta-learners for CATE, 2019

### Week 2 — Medical Question Answering

- [Labeling methods and dataset](https://arxiv.org/abs/1901.07031) — CheXpert (Irvin et al., 2019)
- [Huggingface transformers library](https://github.com/huggingface/transformers)
- [BERT paper](https://arxiv.org/abs/1810.04805) — Devlin et al., 2018
- [Question answering data set (used for example)](https://rajpurkar.github.io/SQuAD-explorer/) — SQuAD

### Week 3 — ML Interpretation

- [Grad-CAM](https://arxiv.org/pdf/1610.02391.pdf) — Selvaraju et al., 2017
- [Random forests + permutation importance](https://www.stat.berkeley.edu/~breiman/randomforest2001.pdf) — Breiman, “Random Forests”, *Machine Learning*, 45(1), 5–32, 2001
- [Shapley importance](https://www.nature.com/articles/s42256-019-0138-9) — Lundberg et al., *Nat Mach Intell*, 2020
- [Clinical note example for question answering](https://www.mtsamples.com/site/pages/sample.asp?Type=6-Cardiovascular%20/%20Pulmonary&Sample=1597-Abnormal%20Echocardiogram) — MTSamples abnormal echocardiogram note

### Course links

- [AI for Medical Treatment — Course home](https://www.coursera.org/learn/ai-for-medical-treatment)
- [Specialization outline](https://www.deeplearning.ai/specializations/ai-for-medicine)
