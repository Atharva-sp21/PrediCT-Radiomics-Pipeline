# PrediCT GSoC 2026 Evaluation: Radiomics & Phenotyping

This repository contains the implementation for the Google Summer of Code (GSoC) 2026 applicant evaluation tasks for the **PrediCT** project. Specifically, this repository fulfills the **Common Task (Data Preprocessing)** and **Project 2 (Radiomics & Phenotyping)**.

The goal of this pipeline is to establish a robust data loading mechanism for the Stanford COCA dataset and to extract, analyze, and visualize radiomic features from cardiac CT scans to identify clinical phenotypes correlated with Agatston scores.

## 🎯 Objectives Met

### 1. Common Task: COCA Dataset Preprocessing
A tailored data pipeline designed for medical imaging compatibility.
* **HU Windowing:** Implemented Hounsfield Unit (HU) windowing specifically optimized for cardiac CT scans to isolate relevant anatomical structures and calcium deposits.
* **Data Splitting:** Engineered a stratified train/validation/test split to ensure balanced class distributions across the dataset.
* **Pipeline Integration:** Created an efficient data loader capable of handling volumetric medical data (DICOM/NIfTI) seamlessly.

### 2. Project 2: Feature Extraction & Statistical Analysis
An end-to-end radiomics pipeline leveraging `PyRadiomics` to correlate image phenotypes with clinical outcomes.
* **Feature Extraction:** Extracted comprehensive feature sets including:
  * **Shape:** Sphericity, Surface-to-Volume Ratio, Maximum 3D Diameter.
  * **Texture (GLCM, GLSZM, GLRLM):** Contrast, Correlation, Small/Large Area Emphasis, Run Percentage, and more.
* **Statistical Analysis:** Calculated clinical Agatston scores and performed robust statistical tests (Spearman correlation, Kruskal-Wallis) to identify features significantly associated with calcium severity categories (0, 1-99, 100-399, ≥400).
* **Phenotype Characterization:** Generated unsupervised clustering visualizations (UMAP/t-SNE) to discover hidden calcium phenotypes within the dataset.

## 📂 Repository Structure

```text
GSOC/
├── data/                           # Stanford COCA dataset (Raw & Processed)
├── src/
│   ├── generate_data.py            # [Common Task] Preprocessing, HU windowing, and data loaders
│   ├── extract_features.py         # [Project 2] PyRadiomics extraction script
│   └── analyze_stats.py            # [Project 2] Statistical correlation, p-values, and visualization
├── calcium_phenotypes_umap.png     # Output: UMAP clustering of patient phenotypes
├── correlation_matrix.png          # Output: Spearman correlations of radiomic features
├── feature_importance.png          # Output: Visualized significance of extracted metrics
├── radiomics_features.csv          # Output: Raw extracted PyRadiomics dataset
└── README.md                       # Project documentation
