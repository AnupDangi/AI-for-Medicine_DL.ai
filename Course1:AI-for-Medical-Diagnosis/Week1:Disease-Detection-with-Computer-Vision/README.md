# Course 1 · Week 1 — Disease Detection with Computer Vision

> Build a deep-learning model that reads chest X-rays and detects up to 14 pathologies at once.
> This week is all about the *practical problems* that make medical imaging hard: messy data,
> heavy class imbalance, tiny training sets, and sneaky data leakage.

---

## Learning objectives

By the end of this week you should be able to:

1. Explore and pre-process a real-world medical image dataset.
2. Explain why medical datasets are usually **multi-label** and **class-imbalanced**.
3. Derive and implement a **weighted cross-entropy loss** to counter class imbalance.
4. Use **transfer learning** with a pre-trained **DenseNet-121** to classify X-rays.
5. Detect and remove **patient overlap / data leakage** between train/val/test splits.
6. Evaluate with **AUROC** and interpret model attention with **Grad-CAM**.

---

## 1. The problem: multi-label chest X-ray classification

The dataset is the NIH **ChestX-ray14** set: frontal chest X-rays, each labeled for the presence
or absence of 14 conditions (e.g. *Cardiomegaly, Edema, Mass, Pneumonia, Effusion*).

- This is **multi-label**, not multi-class: a single X-ray can show *several* diseases at once,
  so each disease gets its own independent binary output (sigmoid, not softmax).
- Each output neuron answers one yes/no question → we use **binary cross-entropy per label**.

### Image pre-processing (Keras `ImageDataGenerator`)
- **Standardization**: subtract the mean and divide by the standard deviation of the *training* set
  so pixels are centered around 0 with unit variance. Crucially, the *same* training statistics are
  reused on validation/test data (never recompute on test data — that leaks information).
- Images are resized (e.g. 320×320) and the single grayscale channel is repeated to 3 channels so a
  model pre-trained on natural RGB images (ImageNet) can be reused.

> Labs: `C1_W1_Lab_1_data_exploration_and_image_preprocessing.ipynb`

---

## 2. Class imbalance and the weighted loss

In medical data, the "disease present" class is usually **rare**. If 98% of images are negative for a
condition, a lazy model that always predicts "negative" gets 98% accuracy but is useless.

### Why plain cross-entropy fails
Binary cross-entropy for one label:

```
L = -( y · log(ŷ) + (1 - y) · log(1 - ŷ) )
```

When negatives vastly outnumber positives, the **sum** of the loss is dominated by the negative
term, so the model is barely penalized for missing the rare positives.

### The fix: weighted loss
Weight each class by the frequency of the *opposite* class, so both classes contribute equally:

```
w_pos = num_negative / total
w_neg = num_positive / total

L = -( w_pos · y · log(ŷ) + w_neg · (1 - y) · log(1 - ŷ) )
```

Now a batch that is mostly negative still gives the positive examples enough "voice" during training.
For multi-label problems you compute these weights **per column (per disease)** and average the loss
over all labels.

> Labs: `C1_W1_Lab_2_counting_labels_and_weighted_loss_function.ipynb`

---

## 3. The model: DenseNet-121 + transfer learning

**DenseNet** (Densely Connected Convolutional Network) connects each layer to *every* subsequent
layer within a block. Instead of adding feature maps (like ResNet), it **concatenates** them:

- Encourages **feature reuse** → fewer parameters for the same accuracy.
- Improves gradient flow → easier to train very deep networks.

### Transfer learning recipe
1. Load DenseNet-121 pre-trained on ImageNet (keeps the learned edge/texture detectors).
2. Remove the original classifier head.
3. Add a global average pooling layer + a new dense layer with **14 sigmoid outputs**.
4. Fine-tune on chest X-rays with the weighted loss.

This lets us train a strong model even though medical datasets are far smaller than ImageNet.

> Labs: `C1_W1_Lab_3_densenet.ipynb`

---

## 4. Data leakage & patient overlap

**The single most important "gotcha" in medical ML.** A patient often has multiple X-rays. If images
of the *same patient* land in both the training and test sets, the model can memorize patient-specific
features and your test score becomes falsely optimistic.

### How to prevent it
- Split **by patient ID**, not by image, so all images from one patient stay in a single split.
- Sanity-check overlap: extract the set of patient IDs in each split and confirm the intersection is
  empty (`set(train_ids) ∩ set(val_ids) == ∅`).

> Labs: `C1_W1_Lab_4_patient_overlap_and_data_leakage.ipynb`

---

## 5. Evaluation & interpretation

- **AUROC (Area Under the ROC Curve)**: the standard metric here. It measures how well the model ranks
  a random positive above a random negative, independent of a specific threshold — computed per disease.
- **Grad-CAM (Gradient-weighted Class Activation Mapping)**: produces a heatmap over the X-ray showing
  *where* the model looked to make its decision. Essential for clinical trust and debugging.

---

## Assignment

**`C1W1_Assignment.ipynb` — Chest X-Ray Medical Diagnosis with Deep Learning**

You will: prevent data leakage, build train/valid/test generators, compute class frequencies, implement
the weighted loss, fine-tune DenseNet-121, then evaluate with AUROC and visualize with Grad-CAM.

---

## Key terms cheat-sheet

| Term | Meaning |
|------|---------|
| Multi-label | One example can belong to several classes simultaneously |
| Class imbalance | One class massively outnumbers another |
| Weighted loss | Loss that up-weights the rare class to balance contributions |
| Transfer learning | Reuse a model pre-trained on a large dataset for a new task |
| DenseNet | CNN where layers concatenate outputs of all previous layers |
| Data leakage | Test information "leaking" into training, inflating scores |
| AUROC | Threshold-independent ranking metric for classifiers |
| Grad-CAM | Heatmap explaining which pixels drove a prediction |

---

## Sources & further reading

- [AI for Medical Diagnosis — Course home](https://www.coursera.org/learn/ai-for-medical-diagnosis)
- [Specialization outline](https://www.deeplearning.ai/specializations/ai-for-medicine)
- [ChestX-ray14 dataset (Wang et al., 2017)](https://arxiv.org/abs/1705.02315)
- [CheXNet (Rajpurkar et al., 2017)](https://arxiv.org/abs/1711.05225)
- [DenseNet paper (Huang et al., 2017)](https://arxiv.org/abs/1608.06993)
- [Grad-CAM paper (Selvaraju et al., 2017)](https://arxiv.org/abs/1610.02391)
- [Keras `ImageDataGenerator` docs](https://keras.io/api/preprocessing/image/)
