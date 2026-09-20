# Cover Letter for Journal Submission

**To:**  
The Editor-in-Chief  
*IEEE Journal of Biomedical and Health Informatics (JBHI)*  
*(Alternative Target: Computers in Biology and Medicine / npj Digital Medicine)*  

**Date:** September 20, 2026  

**Subject:** Submission of Original Research Article titled *"Faithful by Design: A Cross-Modal Concept Bottleneck Framework for Trustworthy Multimodal Clinical Decision Support"*

Dear Editor-in-Chief,

We are pleased to submit our original research manuscript titled **"Faithful by Design: A Cross-Modal Concept Bottleneck Framework for Trustworthy Multimodal Clinical Decision Support"** for consideration for publication as a Regular Paper in the *IEEE Journal of Biomedical and Health Informatics*.

### 1. Clinical Context and Problem Statement
Deploying deep learning in acute clinical decision support (such as triage and mortality risk prediction in intensive care units) requires models to fuse heterogeneous data streams—combining high-dimensional electronic health record (EHR) physiological vitals and laboratory analytes with diagnostic medical imaging (chest radiographs). While unconstrained black-box multimodal networks achieve impressive statistical discrimination, their clinical deployment has repeatedly stalled due to safety concerns and a lack of interpretability. 

Existing clinical AI systems rely almost exclusively on post-hoc explainability methods (e.g., Grad-CAM, Integrated Gradients, SHAP, and LIME). However, recent foundational studies have shown that post-hoc explainers frequently suffer from a critical **faithfulness-plausibility gap**: they produce visual and numeric explanations that appear plausible to human clinicians while failing to reflect the actual decision mechanisms computed by the neural network. In high-stakes medicine, unfaithful explanations exacerbate automation bias and increase clinical alarm fatigue.

### 2. Scientific Contributions of This Work
To address this challenge, we introduce the **Cross-Modal Concept Bottleneck Model (XM-CBM)**, an intrinsically interpretable architecture that is faithful by design:
1. **Inherently Interpretable Multimodal Architecture:** XM-CBM forces all information between tabular EHR and chest radiographs to pass through a human-understandable concept bottleneck, followed by an unconstrained linear predictor that guarantees direct, mathematically transparent attributions.
2. **Faithfulness-Constrained Optimization:** We introduce an explicit empirical faithfulness loss ($\mathcal{L}_{\text{faith}}$) that penalizes any discrepancy between reported concept attribution and observed prediction drops under intervention, alongside a cross-modal alignment penalty.
3. **Rigorous Clinical Benchmark on 12,847 Patients:** Benchmarking against six uni- and multimodal baseline models on matched adult ICU cohorts from MIMIC-IV and MIMIC-CXR, XM-CBM achieves competitive diagnostic performance (AUROC $0.854 \pm 0.012$, AUPRC $0.593 \pm 0.024$, ECE $0.029 \pm 0.006$) while reducing Deletion-AUC by **50.5%**, improving Insertion-AUC by **20.8%**, reducing Explanation Infidelity by **76.2%**, and doubling explanation robustness under adversarial and Gaussian perturbations.

### 3. Declarations
- **Originality:** This manuscript represents original work and has not been published previously, nor is it currently under consideration for publication elsewhere.
- **Ethics and Data Integrity:** All human patient data were obtained from the de-identified MIMIC-IV and MIMIC-CXR databases under PhysioNet Credentialed Data Use Agreements in compliance with HIPAA Safe Harbor guidelines.
- **Reproducibility:** The complete source code, synthetic demonstration pipeline, PyTorch data loaders, training scripts, and evaluation suites are publicly available in an open-access GitHub repository: `https://github.com/mtechbro94/Multimodal-Explainable-AI-for-Clinical-Decision-Support`.
- **Conflicts of Interest:** The authors declare that they have no competing financial or commercial conflicts of interest.

### 4. Suggested Independent Expert Reviewers
1. **Prof. Cynthia Rudin** (Duke University) — Expert in inherently interpretable machine learning and concept-based models (`cynthia.rudin@duke.edu`).
2. **Prof. Marzyeh Ghassemi** (Massachusetts Institute of Technology) — Expert in machine learning for healthcare, clinical datasets (MIMIC), and clinical auditability (`mghassem@mit.edu`).
3. **Prof. Been Kim** (Google DeepMind) — Pioneer in concept activation vectors and sanity checks for saliency methods (`beenkim@google.com`).
4. **Prof. Matthew P. Lungren** (Stanford University / Nuance) — Clinical radiologist and expert in multimodal clinical AI and chest radiograph analysis (`mlungren@stanford.edu`).

Thank you for your time, consideration, and management of the peer-review process. We look forward to hearing from you.

Sincerely,

**Aaqib Rashid Mir**  
Department of Computer Science and Engineering  
Chandigarh University, Mohali, Punjab 140413, India  
*Corresponding Author Email: mtechbro94@gmail.com*  
*GitHub: https://github.com/mtechbro94/Multimodal-Explainable-AI-for-Clinical-Decision-Support*
