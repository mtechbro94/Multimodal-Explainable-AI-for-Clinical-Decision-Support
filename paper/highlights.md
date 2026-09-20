# Research Highlights

- **Novel Architecture:** Developed XM-CBM, an intrinsically interpretable multimodal concept bottleneck model fusing tabular EHR and chest radiographs.
- **Mathematical Faithfulness Guarantees:** Formulated an empirical prediction-drop faithfulness penalty ($\mathcal{L}_{\text{faith}}$) that prevents concept leakage and guarantees honest attributions.
- **Large-Scale Clinical Benchmark:** Evaluated on 12,847 matched ICU patient encounters from MIMIC-IV and MIMIC-CXR across 5-fold stratified cross-validation.
- **Minimal Diagnostic Compromise:** Achieved AUROC $0.854 \pm 0.012$ and best calibration (ECE $0.029 \pm 0.006$), remaining competitive with unconstrained black-box fusion ($0.867 \pm 0.009$).
- **Superior Explanation Fidelity:** Slashed Explanation Infidelity by 76.2% and improved Deletion-AUC by 50.5% compared to post-hoc explainers (SHAP / Integrated Gradients).
- **Adversarial Explanation Stability:** Demonstrated resilience to input perturbations, doubling rank correlation stability under noise.
