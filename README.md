# Faithfulness-Constrained Multimodal Explainable AI for Clinical Decision Support

## Project Structure
```
research/
├── src/                    # Core implementation
│   ├── data/              # Synthetic data generation
│   ├── models/            # All model architectures
│   ├── explainability/    # XAI methods
│   └── evaluation/        # Metrics & evaluation
├── notebooks/             # Colab-ready notebooks
├── results/               # Generated results
└── paper/                 # Research manuscript
```

## Quick Start (Google Colab)
1. Upload this project to Google Drive or clone from GitHub
2. Open `notebooks/run_experiments.ipynb` in Colab
3. Enable GPU runtime (Runtime → Change runtime type → T4 GPU)
4. Run all cells

## Local Setup
```bash
pip install -r src/requirements.txt
python src/train.py --config default
python src/evaluate.py --output results/
```

## Citation
If you use this codebase, please cite the accompanying paper.
