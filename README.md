# Prostate MRI Segmentation

This repository is dedicated to training, evaluation, and analysis of **prostate MRI segmentation models**. The project includes a **custom model implementation**, comparison with existing methods, and **XAI (Explainable AI)** techniques for model interpretability.

---

## Dataset

The project uses the **Prostate158** dataset, a curated collection of biparametric 3T prostate MRI studies.

The dataset includes:

* 158 biparametric prostate MRI studies,
* T2-weighted (T2W), DWI sequences, and ADC maps,
* manually annotated masks of prostate anatomical zones and cancerous lesions,
* histopathologic confirmation for each cancerous lesion,
* images resampled to a common orientation and spatial resolution.

The dataset is **not included** in this repository and must be obtained separately according to its license.

---

## Project Scope

The project covers:

* a custom implementation of a prostate MRI segmentation model,
* comparison with existing architectures (e.g., U-Net, nnU-Net),
* quantitative evaluation using standard segmentation metrics,
* application of XAI methods to interpret model predictions.

---

## Repository Structure

```
prostate-segmentation/
├── data/            # local data (not committed)
├── src/             # source code
├── configs/         # experiment configurations
├── scripts/         # training and inference
├── notebooks/       # analysis and visualization
└── README.md
```

---

## Methods

* CNN- and Transformer-based segmentation models,
* comparison with established baselines,
* metrics: Dice, IoU, Precision, Recall,
* XAI techniques (e.g., attention maps, gradient-based methods).

---

## Objective

The objective of this project is to evaluate the performance of a custom prostate MRI segmentation model and analyze its interpretability in comparison with existing approaches.
