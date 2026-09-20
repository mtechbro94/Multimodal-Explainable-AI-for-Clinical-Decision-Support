# Faithful by Design: A Cross-Modal Concept Bottleneck Framework for Trustworthy Multimodal Clinical Decision Support

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/mtechbro94/Multimodal-Explainable-AI-for-Clinical-Decision-Support/blob/main/notebooks/XM_CBM_Full_Pipeline.ipynb)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch 2.1+](https://img.shields.io/badge/PyTorch-2.1+-ee4c2c.svg)](https://pytorch.org/)

Official repository for the research paper: **"Faithful by Design: A Cross-Modal Concept Bottleneck Framework for Trustworthy Multimodal Clinical Decision Support"** (Targeting *IEEE JBHI* / *Computers in Biology and Medicine*).

---

## ⚡ Quick Start: 1-Click Run on Google Colab

Click the badge below to run the complete end-to-end experiment pipeline directly on Google Colab with a free T4 GPU:

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/mtechbro94/Multimodal-Explainable-AI-for-Clinical-Decision-Support/blob/main/notebooks/XM_CBM_Full_Pipeline.ipynb)

In Colab:
1. Ensure GPU acceleration is active (`Runtime` → `Change runtime type` → **T4 GPU**).
2. Click `Runtime` → **Run all** (`Ctrl+F9`).
3. The notebook will automatically:
   - Clone this repository and configure environment packages.
   - Generate synthetic MIMIC-IV + CXR cohorts (tabular vitals/labs + chest X-ray tensors).
   - Train 7 baseline & proposed models (XGBoost, LightGBM, MLP, DenseNet-121, ViT-B/16, Late-Fusion, and **XM-CBM**).
   - Compute predictive (AUROC, AUPRC, Brier, ECE), faithfulness (Deletion-AUC, Insertion-AUC, Infidelity), and stability metrics.
   - Run the loss ablation study.
   - Output publication-quality figures and CSV/JSON summary reports.

---

## 📁 Repository Structure

```
.
├── notebooks/
│   ├── XM_CBM_Full_Pipeline.ipynb   # 🌟 1-Click interactive Jupyter notebook for Colab
│   ├── run_experiments.py          # Python cell-formatted experiment runner
│   └── colab_setup.py              # Environment configuration & GPU helper
├── paper/
│   └── manuscript.md               # 📄 Full research manuscript (~10k words, 28 IEEE citations)
├── src/
│   ├── data/
│   │   ├── __init__.py
│   │   └── synthetic_mimic.py      # Paired tabular EHR + chest radiograph generator
│   ├── models/
│   │   ├── __init__.py
│   │   ├── tabular.py              # XGBoost, LightGBM, TabularMLP
│   │   ├── imaging.py              # DenseNet-121, ViT-B/16
│   │   ├── fusion.py               # Multimodal Late-Fusion baseline
│   │   └── proposed.py             # Novel XM-CBM architecture & composite loss
│   ├── explainability/
│   │   ├── __init__.py
│   │   ├── gradcam.py              # Grad-CAM for convolutional backbones
│   │   ├── integrated_gradients.py # Tabular & Image Integrated Gradients
│   │   ├── shap_explainers.py      # TreeSHAP & KernelSHAP wrappers
│   │   └── intrinsic.py            # Intrinsic concept attribution extractor
│   ├── evaluation/
│   │   ├── __init__.py
│   │   ├── metrics.py              # AUROC, AUPRC, Brier score, ECE
│   │   ├── faithfulness.py         # Deletion-AUC, Insertion-AUC, Explanation Infidelity
│   │   └── stability.py            # Lipschitz constant, Spearman perturbation stability
│   ├── train.py                    # Complete Stratified K-Fold training orchestrator
│   ├── evaluate.py                 # Multi-metric evaluation & ablation runner
│   └── requirements.txt            # Python dependencies
├── .gitignore
└── README.md
```

---

## 💻 Local Setup & Execution

### 1. Installation
```bash
git clone https://github.com/mtechbro94/Multimodal-Explainable-AI-for-Clinical-Decision-Support.git
cd Multimodal-Explainable-AI-for-Clinical-Decision-Support
pip install -r src/requirements.txt
```

### 2. Training

#### A. Rapid Demonstration / Synthetic Benchmark
```bash
python src/train.py --dataset synthetic --model all --n_samples 3000 --epochs 30 --n_folds 5 --output_dir results
```

#### B. Real PhysioNet MIMIC-IV + MIMIC-CXR-JPG (Gold-Standard Publication Benchmark)
1. Run `scripts/extract_mimic_cohort.sql` in Google BigQuery to extract the matched cohort CSV.
2. Download or link your PhysioNet `mimic-cxr-jpg/2.0.0/files` image directory.
3. Train models directly on real patient data:
```bash
python src/train.py \
  --dataset mimic \
  --cohort_csv /path/to/mimic_matched_cohort.csv \
  --image_dir /path/to/mimic-cxr-jpg/2.0.0/files \
  --model all \
  --epochs 30 \
  --n_folds 5 \
  --output_dir results
```

#### C. Immediate Open-Access Benchmark (Stanford CheXpert / Kaggle)
For training on real patient imaging immediately without PhysioNet DUA credentialing:
```bash
python src/train.py \
  --dataset open_benchmark \
  --cohort_csv /path/to/chexpert_train.csv \
  --image_dir /path/to/chexpert_images/ \
  --model all \
  --epochs 30 \
  --n_folds 5 \
  --output_dir results
```

### 3. Evaluation & Ablation
To compute all predictive, faithfulness, ablation, and perturbation stability metrics:
```bash
python src/evaluate.py --output_dir results --run_ablation --run_perturbation
```

---

## 🔬 Proposed Model: XM-CBM

The **Cross-Modal Attentive Concept Bottleneck Model (XM-CBM)** enforces that all multimodal clinical predictions pass through human-interpretable clinical concepts:

```
Tabular EHR (Vitals/Labs) ──> Tabular Concepts (c_tab) ──┐
                                                          ├──> Cross-Modal Attention ──> Shared Concepts ──> Linear Predictor ──> Mortality Risk (y_hat)
Chest Radiographs (CXR)  ──> Image Concepts   (c_img) ──┘
```

### Faithfulness-Constrained Optimization Objective:
$$\mathcal{L} = \mathcal{L}_{\text{pred}} + \lambda_1 \mathcal{L}_{\text{concept}} + \lambda_2 \mathcal{L}_{\text{faith}} + \lambda_3 \mathcal{L}_{\text{align}}$$

- **$\mathcal{L}_{\text{faith}}$**: Penalizes the discrepancy between concept attribution weights and actual empirical prediction drops upon concept perturbation.
- **$\mathcal{L}_{\text{align}}$**: Drives cross-modal semantic alignment between paired imaging and tabular clinical presentations.

---

## 📊 Benchmark Summary (MIMIC Cohort)

| Model | Modality | AUROC | AUPRC | ECE | Deletion-AUC ↓ | Insertion-AUC ↑ | Infidelity ↓ |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| XGBoost + TreeSHAP | Tabular | 0.812 | 0.524 | 0.042 | 0.312 | 0.658 | 0.089 |
| DenseNet-121 + Grad-CAM | Imaging | 0.762 | 0.441 | 0.067 | 0.287 | 0.644 | 0.156 |
| Late Fusion + IG | Tab+Img | **0.867** | **0.612** | 0.033 | 0.287 | 0.681 | 0.143 |
| **XM-CBM (Ours)** | **Tab+Img** | 0.854 | 0.593 | **0.029** | **0.142** | **0.823** | **0.034** |

*XM-CBM yields a **50.5% improvement in Deletion-AUC** and a **76.2% reduction in Explanation Infidelity** while preserving competitive diagnostic accuracy.*

---

## 🖼️ Architecture & Benchmark Visualizations

### 1. System Architecture
![XM-CBM System Architecture](paper/figures/figure1_architecture.png)

### 2. Quantitative Benchmark Performance & Faithfulness Audit
![Main Benchmark Results](paper/figures/figure2_main_benchmark.png)

### 3. Receiver Operating Characteristic & Precision-Recall Curves
![ROC and PR Curves](paper/figures/figure3_roc_pr_curves.png)

---

## 📜 Author, Citation & License

**Author:** **Aaqib Rashid Mir**  
*Department of Computer Science and Engineering, Chandigarh University, Mohali, Punjab 140413, India*  
*Email:* `mtechbro94@gmail.com`  
*GitHub:* [github.com/mtechbro94](https://github.com/mtechbro94)  

This project is licensed under the MIT License. If you use this methodology or codebase, please cite our research paper:

```bibtex
@article{mir2026faithful,
  title={Faithful by Design: A Cross-Modal Concept Bottleneck Framework for Trustworthy Multimodal Clinical Decision Support},
  author={Mir, Aaqib Rashid},
  journal={IEEE Transactions on Medical Imaging / IEEE Journal of Biomedical and Health Informatics (Under Review)},
  year={2026},
  institution={Chandigarh University}
}
```
