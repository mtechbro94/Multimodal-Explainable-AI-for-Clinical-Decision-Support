# Deployment Guide: XM-CBM Clinical Decision Support System

Author: **Aaqib Rashid Mir**  
Affiliation: **Chandigarh University**  
Email: **mtechbro94@gmail.com**  
Repository: [Multimodal-Explainable-AI-for-Clinical-Decision-Support](https://github.com/mtechbro94/Multimodal-Explainable-AI-for-Clinical-Decision-Support)

---

## 1. Project Completion & Deployment Readiness Status

| Component | Status | Readiness Level |
| :--- | :--- | :--- |
| **Research Manuscript & LaTeX** | 100% Complete | Ready for Q1 Journal Submission (*IEEE JBHI*, *Computers in Biology and Medicine*) |
| **Experimental Results & Figures** | 100% Complete | All 6 figures (300 DPI) & 4 CSV result tables generated |
| **Interactive Clinical Web App (`app.py`)** | 100% Complete | Turnkey Streamlit application with test-time concept intervention |
| **Docker Containerization (`Dockerfile`)** | 100% Complete | Production-ready multi-platform container |
| **Public Cloud Deployment** | 100% Ready | Ready for 1-Click Free Deployment (Streamlit Cloud, Hugging Face Spaces) |
| **Hospital Bedside Clinical Deployment** | Research Grade | Requires prospective clinical trial & IRB regulatory clearance before live patient diagnosis |

---

## 2. Option A: Free 1-Click Cloud Deployment via Streamlit Community Cloud (Recommended)

Streamlit Community Cloud is 100% free and hosts your live interactive app directly from your GitHub repository:

1. Visit [share.streamlit.io](https://share.streamlit.io/) and log in with your GitHub account.
2. Click **New App**.
3. Select your repository: `mtechbro94/Multimodal-Explainable-AI-for-Clinical-Decision-Support`.
4. Branch: `main`.
5. Main file path: `app.py`.
6. Click **Deploy!**
   - Streamlit will automatically read `requirements.txt`, install dependencies, and launch your live public URL (e.g. `https://xmcbm-clinical.streamlit.app`) within 2 minutes!

---

## 3. Option B: Free Deployment on Hugging Face Spaces

Hugging Face Spaces provides free GPU/CPU hosting for machine learning models:

1. Create a free account at [huggingface.co](https://huggingface.co/).
2. Click **New Space** -> Choose **Streamlit** (or **Docker**).
3. Name your space (e.g. `XM-CBM-Clinical-Decision-Support`).
4. Set Space to **Public**.
5. Connect your GitHub repository or clone the space and push your files:
   ```bash
   git remote add space https://huggingface.co/spaces/YOUR_USERNAME/XM-CBM-Clinical-Decision-Support
   git push space main
   ```
6. Hugging Face will automatically build and host the live interactive dashboard.

---

## 4. Option C: Local or On-Premise Hospital Docker Deployment

To run the application inside an air-gapped hospital intranet or private server without cloud exposure:

### Step 1: Build Docker Image
```bash
docker build -t xmcbm-clinical-app:latest .
```

### Step 2: Run Container
```bash
docker run -d -p 8501:8501 --name xmcbm-service xmcbm-clinical-app:latest
```

### Step 3: Access Dashboard
Open your web browser and navigate to:
```
http://localhost:8501
```

---

## 5. Option D: Local Python Execution

If you wish to run the app directly on your machine without Docker:

```bash
# Install dependencies
pip install -r requirements.txt

# Launch Streamlit dashboard
streamlit run app.py
```

---

## 6. Clinical & Regulatory Governance Notice

> **Important Regulatory Disclaimer:**
> In accordance with the **EU Artificial Intelligence Act (High-Risk AI Systems)** and **US FDA Software as a Medical Device (SaMD Class II)** regulations:
> - This software and model are published for **research, peer review, and clinical investigational purposes**.
> - It is **not** currently cleared for autonomous diagnostic or therapeutic decisions on living human patients without independent attending physician verification and institutional ethics review (IRB).
> - The interactive counterfactual intervention feature is designed as a cognitive decision-support aid to prevent alarm fatigue and enable clinician auditability.
