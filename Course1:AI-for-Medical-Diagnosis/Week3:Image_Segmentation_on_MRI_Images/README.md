# Course 1 · Week 3 — Image Segmentation on MRI Images

> Move from *"is there a disease?"* (classification) to *"exactly which voxels are tumor?"* (segmentation).
> You'll build a **3D U-Net** to auto-segment brain tumors from multi-modal MRI scans and evaluate it
> with the **Dice coefficient** and **soft Dice loss**.

---

## Learning objectives

1. Understand how **MRI data** is represented as a 3D/4D volume with multiple modalities.
2. Handle huge 3D volumes with **sub-volume (patch) sampling** and standardization.
3. Explain the **U-Net** architecture: contracting path, expanding path, and skip connections.
4. Extend U-Net to **3D** for volumetric segmentation.
5. Measure overlap with the **Dice Similarity Coefficient** and train with **Soft Dice Loss**.
6. Evaluate per-class **sensitivity & specificity** on segmentation output.

---

## 1. What is an MRI, and how is it stored?

An **MRI (Magnetic Resonance Imaging)** scan is a 3D image of the body made of **voxels** (3D pixels).
This dataset (BraTS-style brain tumors) provides **4 modalities** per patient — e.g. FLAIR, T1, T1ce,
T2 — each highlighting different tissue. Think of them as 4 "channels" of the same 3D brain.

- Data shape is roughly `(height, width, depth, channels)`.
- **Image registration** aligns the modalities so the same voxel location means the same anatomy in
  every channel.
- The **label** is a mask assigning each voxel a class (background, edema, enhancing tumor, etc.).

> Labs: `C1_W3_Lab_1_explore_mri_data_and_labels.ipynb`

---

## 2. Working with patches (sub-volumes)

A full MRI volume is too large to fit a 3D network in memory, and most voxels are just background.
Solution: train on smaller **sub-volumes** (patches), e.g. 160×160×16.

- **Random start indices**: sample patches from random positions to cover the volume and augment data.
- **Background ratio filtering**: reject patches that are almost entirely background so the model sees
  enough tumor tissue and doesn't just learn "predict background everywhere."
- **Standardization**: normalize each patch to mean 0, standard deviation 1 (per channel).

> Labs: `C1_W3_Lab_2_extract_a_sub_section.ipynb`

---

## 3. The U-Net architecture

U-Net is *the* workhorse of medical image segmentation. It is shaped like a **"U"**:

```
   Contracting path (encoder)        Expanding path (decoder)
   captures CONTEXT (what)           enables LOCALIZATION (where)
        │                                        ▲
        ▼   ── skip connections (copy) ──►       │
      downsample (max-pool)               upsample (up-conv)
```

- **Contracting/downward path**: repeated conv → conv → max-pool. Spatial size shrinks, feature depth
  grows → the network learns *what* is in the image.
- **Expanding/upward path**: up-convolution (upsampling) back to full resolution → learns *where*.
- **Skip connections**: feature maps from the encoder are **concatenated** to the matching decoder level.
  This restores fine spatial detail lost during downsampling — the key trick that makes U-Net accurate.
- **Final 1×1 convolution** + activation (sigmoid/softmax) produces a per-voxel class probability.

The **"depth"** of the U-Net is how many down/up levels it has. For volumetric MRI we use a **3D U-Net**
where all convolutions and pooling operate over 3 spatial dimensions.

> Labs: `C1_W3_Lab_3_unet_model.ipynb`

---

## 4. Metrics & loss for segmentation

Ordinary accuracy is useless here: a tumor might be <1% of voxels, so "predict all background" scores
99%+. Segmentation uses **overlap-based** metrics instead.

### Dice Similarity Coefficient (DSC)
Measures overlap between prediction `P` and ground truth `G`:
```
Dice = 2 · |P ∩ G| / (|P| + |G|)
```
1.0 = perfect overlap, 0 = none. For multiple classes, average the per-class Dice.

### Soft Dice Loss
Dice uses hard 0/1 masks and isn't differentiable. The **soft** version plugs in the predicted
*probabilities* so it can be optimized by gradient descent:
```
Soft Dice Loss = 1 − (2 · Σ pᵢgᵢ + ε) / (Σ pᵢ² + Σ gᵢ² + ε)
```
Minimizing it directly maximizes overlap and naturally handles class imbalance.

### Per-class sensitivity & specificity
After thresholding, compute sensitivity/specificity **per class** to see, e.g., how much enhancing
tumor the model catches vs. how much healthy tissue it wrongly flags.

---

## Assignment

**`C1W3_Assignment.ipynb` — Brain Tumor Auto-Segmentation for MRI**

Implement `get_sub_volume` (patch sampling), standardization, single- and multi-class Dice coefficients,
soft Dice loss, then load/train a 3D U-Net and evaluate with per-class sensitivity/specificity on full scans.

---

## Broader themes covered this week

- **2D vs 3D segmentation** trade-offs.
- **Data augmentation** for segmentation.
- **External validation**, retrospective vs prospective data, cleaned vs raw data.
- **Algorithmic bias** and how models influence real medical decisions.

---

## Key terms cheat-sheet

| Term | Meaning |
|------|---------|
| Voxel | 3D pixel in a volumetric image |
| Modality | One MRI sequence (FLAIR/T1/T1ce/T2), treated as a channel |
| Registration | Aligning modalities so voxels correspond anatomically |
| Sub-volume / patch | Small 3D crop used for memory-efficient training |
| U-Net | Encoder–decoder CNN with skip connections for segmentation |
| Skip connection | Concatenating encoder features into the decoder |
| Dice coefficient | Overlap metric: 2·intersection / sum of sizes |
| Soft Dice loss | Differentiable Dice using predicted probabilities |

---

## Sources & further reading

- [AI for Medical Diagnosis — Course home](https://www.coursera.org/learn/ai-for-medical-diagnosis)
- [U-Net paper (Ronneberger et al., 2015)](https://arxiv.org/abs/1505.04597)
- [3D U-Net paper (Çiçek et al., 2016)](https://arxiv.org/abs/1606.06650)
- [V-Net & Dice loss (Milletari et al., 2016)](https://arxiv.org/abs/1606.04797)
- [BraTS brain tumor segmentation challenge](https://www.med.upenn.edu/cbica/brats2020/)
- [NiBabel — reading neuroimaging data in Python](https://nipy.org/nibabel/)
