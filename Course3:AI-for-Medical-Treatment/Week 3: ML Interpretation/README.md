# Course 3 · Week 3 — ML Interpretation

> Clinical models only get adopted if clinicians can **trust and understand** them. This week you
> explain **deep imaging models** with **Grad-CAM** heatmaps and **tabular models** with
> **permutation importance** and **Shapley (SHAP)** values — revisiting Course 1 X-rays and Course 2
> risk forests.

---

## Learning objectives

1. Extract intermediate CNN layer activations and class gradients with Keras.
2. Implement **Grad-CAM** to highlight image regions driving a prediction.
3. Measure **permutation feature importance** without retraining the model.
4. Read **SHAP** force / summary / dependence plots for local and global explanations.
5. Contrast global importance (permutation) with local attributions (SHAP).

---

## 0. Foundations from the labs

- **Lab 1 — `C3_W3_Lab_1_permutation_method.ipynb`**
  Intuition for permutation importance on a Course 2 risk model: shuffle one feature, re-score,
  and see how much performance drops.

- **Lab 2 — `C3_W3_Lab_2_intro_to_gradcam.ipynb`**
  Grad-CAM Part 1: why CNNs need visual explanations, and how to pull **activations** from a chosen
  convolutional layer with the Keras functional API.

- **Lab 3 — `C3_W3_Lab_3_gradcam_continuation.ipynb`**
  Grad-CAM Part 2: compute **gradients** of the class score w.r.t. that layer's output — the "Grad"
  half of Grad-CAM — ready to combine into a heatmap in the assignment.

---

## 1. Grad-CAM (deep learning)

**Grad-CAM** (Gradient-weighted Class Activation Mapping) answers: *which pixels mattered for this
class?*

Pipeline:

1. Forward pass → take feature maps from a late convolutional layer.
2. Backprop the **class score** → gradients w.r.t. those maps.
3. Globally average-pool gradients → channel weights.
4. Weighted sum of maps → ReLU → upsample → overlay on the X-ray.

Use cases in medicine: error analysis, discovering markers, checking the model isn't looking at
spurious artifacts (tubes, text overlays, laterality markers).

> Assignment exercise: `grad_cam` (+ multi-label visualization for Cardiomegaly / Mass / Edema)

---

## 2. Permutation importance (model-agnostic)

For feature $x$:

$$
\text{importance}(x) = \text{perf}(\text{original}) - \mathbb{E}\big[\text{perf}(x\text{ shuffled})\big]
$$

Shuffle only column $x$, keep the fitted model fixed, and average over many shuffles. Large drop ⇒
important feature. Model-agnostic and global (one score per feature for the whole dataset).

> Assignment exercises: `permute_feature`, `permutation_importance`

---

## 3. Shapley / SHAP values

**Shapley values** from cooperative game theory attribute a prediction fairly among features so that
attributions **sum to the prediction** (relative to a baseline).

- **Local:** force plots show why *this patient* got *this* risk.
- **Global:** summary plots aggregate |SHAP| across patients.
- **Interactions:** dependence plots color one feature's SHAP by another (e.g. age × sex).

Tree ensembles admit fast SHAP approximations — applied here to the Course 2 random forest.

---

## Notebooks this week

| Notebook | Role |
|----------|------|
| `C3_W3_Lab_1_permutation_method.ipynb` | Permutation importance demo |
| `C3_W3_Lab_2_intro_to_gradcam.ipynb` | Activations for Grad-CAM |
| `C3_W3_Lab_3_gradcam_continuation.ipynb` | Gradients for Grad-CAM |
| `C3W3_Assignment.ipynb` | Grad-CAM + permutation + SHAP |

---

## Key terms

| Term | Meaning |
|------|---------|
| Grad-CAM | Heatmap of class-relevant image regions via gradients × activations |
| Permutation importance | Drop in metric when a feature is shuffled |
| SHAP / Shapley | Game-theoretic feature attributions that sum to the prediction |
| Force plot | Local SHAP visualization for one example |
| Dependence plot | SHAP vs feature value, optionally colored by another feature |
| Global vs local | Dataset-level ranking vs per-prediction explanation |

---

## Citations (Week 3)

- [Grad-CAM](https://arxiv.org/pdf/1610.02391.pdf) — Selvaraju et al., 2017
- [Random forests + permutation importance](https://www.stat.berkeley.edu/~breiman/randomforest2001.pdf) — Breiman, “Random Forests”, *Machine Learning*, 45(1), 5–32, 2001
- [Shapley importance](https://www.nature.com/articles/s42256-019-0138-9) — Lundberg et al., *Nat Mach Intell*, 2020
- [Clinical note example for question answering](https://www.mtsamples.com/site/pages/sample.asp?Type=6-Cardiovascular%20/%20Pulmonary&Sample=1597-Abnormal%20Echocardiogram) — MTSamples

### Additional references

- [AI for Medical Treatment — Course home](https://www.coursera.org/learn/ai-for-medical-treatment)
- [SHAP documentation](https://shap.readthedocs.io/en/latest/)
- [A Unified Approach to Interpreting Model Predictions (Lundberg & Lee, 2017)](https://arxiv.org/abs/1705.07874)
