import argparse
import os
import sys
import json
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Subset
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from data.synthetic_mimic import SyntheticMIMICDataset
from models.tabular import TabularMLP, XGBoostWrapper, LightGBMWrapper
from models.imaging import DenseNet121Classifier, ViTClassifier
from models.fusion import LateFusionModel
from models.proposed import CrossModalConceptBottleneck, XMCBMLoss
from explainability.gradcam import GradCAM
from explainability.integrated_gradients import IntegratedGradients
from explainability.shap_explainers import TreeSHAPExplainer, KernelSHAPExplainer
from explainability.intrinsic import IntrinsicConceptExplainer
from evaluation.metrics import compute_auroc, compute_auprc, compute_brier_score, compute_ece, compute_all_metrics
from evaluation.faithfulness import compute_deletion_auc, compute_insertion_auc, compute_infidelity, compute_all_faithfulness
from evaluation.stability import estimate_lipschitz_constant, explanation_sensitivity, topk_stability
from train import collate_fn, set_seed, validate_neural


def load_model(model_name, model_path, tab_dim, device):
    """Load a trained model from checkpoint."""
    if model_name in ['xgboost', 'lightgbm']:
        model_class = XGBoostWrapper if model_name == 'xgboost' else LightGBMWrapper
        model = model_class()
        return model
    elif model_name == 'mlp':
        model = TabularMLP(input_dim=tab_dim).to(device)
    elif model_name == 'densenet':
        model = DenseNet121Classifier().to(device)
    elif model_name == 'vit':
        model = ViTClassifier().to(device)
    elif model_name == 'late_fusion':
        tab_encoder = TabularMLP(input_dim=tab_dim)
        img_encoder = DenseNet121Classifier()
        model = LateFusionModel(tab_encoder, img_encoder).to(device)
    elif model_name == 'xm_cbm':
        model = CrossModalConceptBottleneck(tab_input_dim=tab_dim).to(device)
    else:
        raise ValueError(f"Unknown model: {model_name}")

    if model_path and os.path.exists(model_path):
        model.load_state_dict(torch.load(model_path, map_location=device, weights_only=True))
    model.eval()
    return model


def get_explainer(model, model_name, background_data=None):
    """Return the appropriate explainer for a given model type."""
    if model_name in ['xgboost', 'lightgbm']:
        return TreeSHAPExplainer(model.get_booster())
    elif model_name == 'mlp':
        model_fn = lambda x: model(torch.tensor(x, dtype=torch.float32)).detach().numpy()
        return KernelSHAPExplainer(model_fn, background_data)
    elif model_name == 'densenet':
        return GradCAM(model, model.get_cam_target_layer())
    elif model_name == 'vit':
        return IntegratedGradients(model)
    elif model_name == 'late_fusion':
        return IntegratedGradients(model)
    elif model_name == 'xm_cbm':
        return IntrinsicConceptExplainer(model)
    return None


def make_model_fn(model, model_name, device, fixed_img=None, fixed_tab=None):
    """Create a callable model function for faithfulness evaluation."""
    model.eval()

    def fn(x):
        with torch.no_grad():
            x_t = torch.tensor(x, dtype=torch.float32).to(device) if not torch.is_tensor(x) else x.to(device)
            if model_name in ['xgboost', 'lightgbm']:
                if hasattr(model, 'predict_proba'):
                    return model.predict_proba(x if not torch.is_tensor(x) else x.cpu().numpy())
            elif model_name in ['late_fusion', 'xm_cbm']:
                if fixed_img is not None:
                    img = fixed_img.to(device).expand(x_t.shape[0], -1, -1, -1) if x_t.dim() == 2 else fixed_img.to(device)
                    if model_name == 'xm_cbm':
                        logits, _ = model(x_t, img)
                    else:
                        logits = model(x_t, img)
                elif fixed_tab is not None:
                    tab = fixed_tab.to(device).expand(x_t.shape[0], -1) if x_t.dim() > 2 else fixed_tab.to(device)
                    if model_name == 'xm_cbm':
                        logits, _ = model(tab, x_t)
                    else:
                        logits = model(tab, x_t)
                else:
                    return 0.5
                return torch.sigmoid(logits).cpu().numpy().flatten()
            else:
                logits = model(x_t)
                return torch.sigmoid(logits).cpu().numpy().flatten()
    return fn


def evaluate_predictions(y_true, y_prob):
    """Compute all predictive performance metrics."""
    return compute_all_metrics(y_true, y_prob)


def evaluate_faithfulness_for_samples(model_fn, samples, attributions, n_steps=20, n_perturbations=50):
    """Compute faithfulness metrics over a batch of samples."""
    del_aucs, ins_aucs, infids = [], [], []

    for i in range(len(samples)):
        x = samples[i:i+1] if torch.is_tensor(samples) else samples[i:i+1]
        attr = attributions[i:i+1] if torch.is_tensor(attributions) else attributions[i:i+1]
        try:
            del_auc = compute_deletion_auc(model_fn, x, attr, n_steps)
            ins_auc = compute_insertion_auc(model_fn, x, attr, n_steps)
            infid = compute_infidelity(model_fn, x, attr, n_perturbations)
            del_aucs.append(del_auc)
            ins_aucs.append(ins_auc)
            infids.append(infid)
        except Exception:
            continue

    return {
        'deletion_auc': (np.mean(del_aucs), np.std(del_aucs)) if del_aucs else (0, 0),
        'insertion_auc': (np.mean(ins_aucs), np.std(ins_aucs)) if ins_aucs else (0, 0),
        'infidelity': (np.mean(infids), np.std(infids)) if infids else (0, 0),
    }


def evaluate_stability_for_samples(explain_fn, samples, n_samples=20):
    """Compute explanation stability metrics over samples."""
    lip_maxes, lip_means = [], []
    sensitivities = {0.01: [], 0.05: [], 0.1: []}
    topk_jaccards = []

    for i in range(min(n_samples, len(samples))):
        x = samples[i:i+1] if torch.is_tensor(samples) else samples[i:i+1]
        try:
            lip_max, lip_mean = estimate_lipschitz_constant(explain_fn, x, n_perturbations=10, epsilon=0.01)
            lip_maxes.append(lip_max)
            lip_means.append(lip_mean)

            sens = explanation_sensitivity(explain_fn, x, sigmas=[0.01, 0.05, 0.1])
            for sigma, corr in sens.items():
                sensitivities[sigma].append(corr)

            topk = topk_stability(explain_fn, x, k=5, n_perturbations=10, epsilon=0.01)
            topk_jaccards.append(topk)
        except Exception:
            continue

    return {
        'lipschitz_max': (np.mean(lip_maxes), np.std(lip_maxes)) if lip_maxes else (0, 0),
        'lipschitz_mean': (np.mean(lip_means), np.std(lip_means)) if lip_means else (0, 0),
        'sensitivity': {s: (np.mean(v), np.std(v)) for s, v in sensitivities.items() if v},
        'topk_jaccard': (np.mean(topk_jaccards), np.std(topk_jaccards)) if topk_jaccards else (0, 0),
    }


def run_full_evaluation(trained_models, dataset, n_folds, device, output_dir):
    """
    Run the complete evaluation pipeline across all models and folds.
    Returns a DataFrame with all metrics.
    """
    labels = np.array([dataset[i]['label'].item() for i in range(len(dataset))])
    skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=42)

    all_results = []

    for model_name, model in trained_models.items():
        fold_metrics = []

        for fold, (train_idx, val_idx) in enumerate(skf.split(np.zeros(len(labels)), labels)):
            val_subset = Subset(dataset, val_idx)
            val_loader = DataLoader(val_subset, batch_size=64, shuffle=False, collate_fn=collate_fn)

            is_mm = model_name in ['late_fusion', 'xm_cbm']
            y_true, y_prob = validate_neural(model, val_loader, device, is_mm)

            pred_metrics = evaluate_predictions(y_true, y_prob.flatten())
            fold_metrics.append(pred_metrics)

        means = {k: np.mean([f[k] for f in fold_metrics]) for k in fold_metrics[0]}
        stds = {k: np.std([f[k] for f in fold_metrics]) for k in fold_metrics[0]}

        row = {'model': model_name}
        for k in means:
            row[f'{k}_mean'] = means[k]
            row[f'{k}_std'] = stds[k]
        all_results.append(row)

    df = pd.DataFrame(all_results)
    df.to_csv(os.path.join(output_dir, 'benchmark_results.csv'), index=False)
    return df


def generate_ablation_results(dataset, device, output_dir, n_folds=5):
    """
    Train XM-CBM variants with different loss configurations and evaluate.
    """
    tab_dim = dataset.get_tabular_dim()
    configs = [
        ('XM-CBM (full)', {'lambda_concept': 0.1, 'lambda_faith': 0.5, 'lambda_align': 0.2}),
        ('w/o L_faith', {'lambda_concept': 0.1, 'lambda_faith': 0.0, 'lambda_align': 0.2}),
        ('w/o L_align', {'lambda_concept': 0.1, 'lambda_faith': 0.5, 'lambda_align': 0.0}),
        ('w/o L_concept', {'lambda_concept': 0.0, 'lambda_faith': 0.5, 'lambda_align': 0.2}),
    ]

    results = []
    labels = np.array([dataset[i]['label'].item() for i in range(len(dataset))])
    skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=42)

    for name, loss_params in configs:
        fold_aucs = []
        for fold, (train_idx, val_idx) in enumerate(skf.split(np.zeros(len(labels)), labels)):
            model = CrossModalConceptBottleneck(tab_input_dim=tab_dim).to(device)
            loss_fn = XMCBMLoss(**loss_params)

            train_loader = DataLoader(Subset(dataset, train_idx), batch_size=64, shuffle=True, collate_fn=collate_fn)
            val_loader = DataLoader(Subset(dataset, val_idx), batch_size=64, shuffle=False, collate_fn=collate_fn)

            from train import train_neural_model
            train_neural_model(model, train_loader, val_loader, epochs=10, lr=1e-4,
                             device=device, model_name=f'ablation_{name}_f{fold}',
                             output_dir=output_dir, patience=5, loss_fn=loss_fn)

            y_true, y_prob = validate_neural(model, val_loader, device, is_multimodal=True)
            fold_aucs.append(roc_auc_score(y_true, y_prob))

        results.append({
            'variant': name,
            'auroc_mean': np.mean(fold_aucs),
            'auroc_std': np.std(fold_aucs),
        })

    df = pd.DataFrame(results)
    df.to_csv(os.path.join(output_dir, 'ablation_results.csv'), index=False)
    return df


def generate_perturbation_results(model, explainer_fn, x_samples, device, sigmas=[0.01, 0.05, 0.1]):
    """
    Evaluate explanation stability under input perturbations.
    """
    results = []

    for sigma in sigmas:
        original_attrs = []
        perturbed_attrs = []

        for i in range(min(50, len(x_samples))):
            x = x_samples[i:i+1]
            try:
                attr_orig = explainer_fn(x)
                noise = torch.randn_like(torch.tensor(x, dtype=torch.float32)) * sigma if not torch.is_tensor(x) else torch.randn_like(x) * sigma
                x_pert = (x + noise.numpy()) if not torch.is_tensor(x) else x + noise
                attr_pert = explainer_fn(x_pert)

                original_attrs.append(np.array(attr_orig).flatten())
                perturbed_attrs.append(np.array(attr_pert).flatten())
            except Exception:
                continue

        if original_attrs:
            from scipy.stats import spearmanr
            rhos = [spearmanr(o, p).correlation for o, p in zip(original_attrs, perturbed_attrs) if not np.isnan(spearmanr(o, p).correlation)]

            top_k = 5
            jaccards = []
            for o, p in zip(original_attrs, perturbed_attrs):
                top_orig = set(np.argsort(-np.abs(o))[:top_k])
                top_pert = set(np.argsort(-np.abs(p))[:top_k])
                jaccards.append(len(top_orig & top_pert) / len(top_orig | top_pert))

            results.append({
                'sigma': sigma,
                'spearman_rho_mean': np.mean(rhos) if rhos else 0,
                'spearman_rho_std': np.std(rhos) if rhos else 0,
                'topk_jaccard_mean': np.mean(jaccards),
                'topk_jaccard_std': np.std(jaccards),
            })

    df = pd.DataFrame(results)
    df.to_csv(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'results', 'perturbation_results.csv'), index=False)
    return df


def print_results_table(results_df, title):
    """Pretty-print a results table."""
    print(f"\n{'='*60}")
    print(f"  {title}")
    print(f"{'='*60}")
    try:
        print(results_df.to_markdown(index=False))
    except ImportError:
        print(results_df.to_string(index=False))
    print()


def main():
    parser = argparse.ArgumentParser(description='Multimodal XAI Evaluation Pipeline')
    parser.add_argument('--models_dir', type=str, default='../results/checkpoints')
    parser.add_argument('--output_dir', type=str, default='../results')
    parser.add_argument('--n_folds', type=int, default=5)
    parser.add_argument('--n_samples', type=int, default=3000)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--run_ablation', action='store_true')
    parser.add_argument('--run_perturbation', action='store_true')
    args = parser.parse_args()

    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    dataset = SyntheticMIMICDataset(n_samples=args.n_samples)
    tab_dim = dataset.get_tabular_dim()
    os.makedirs(args.output_dir, exist_ok=True)

    # Load or create models for evaluation
    models_dict = {}
    model_configs = {
        'mlp': lambda: TabularMLP(input_dim=tab_dim).to(device),
        'densenet': lambda: DenseNet121Classifier().to(device),
        'vit': lambda: ViTClassifier().to(device),
        'xm_cbm': lambda: CrossModalConceptBottleneck(tab_input_dim=tab_dim).to(device),
    }

    for mname, constructor in model_configs.items():
        ckpt = os.path.join(args.models_dir, f'{mname}_best.pt')
        model = constructor()
        if os.path.exists(ckpt):
            model.load_state_dict(torch.load(ckpt, map_location=device, weights_only=True))
            print(f"Loaded checkpoint for {mname}")
        else:
            print(f"No checkpoint found for {mname}, using randomly initialized weights")
        model.eval()
        models_dict[mname] = model

    # Run full evaluation
    print("\n--- Running Full Evaluation ---")
    df_eval = run_full_evaluation(models_dict, dataset, args.n_folds, device, args.output_dir)
    print_results_table(df_eval, "Benchmark Results (Predictive Metrics)")

    # Ablation study
    if args.run_ablation:
        print("\n--- Running Ablation Study ---")
        df_abl = generate_ablation_results(dataset, device, args.output_dir, args.n_folds)
        print_results_table(df_abl, "Ablation Study Results")

    # Perturbation analysis
    if args.run_perturbation:
        print("\n--- Running Perturbation Analysis ---")
        xm_cbm = models_dict.get('xm_cbm')
        if xm_cbm:
            explainer = IntrinsicConceptExplainer(xm_cbm)
            samples_tab = torch.stack([dataset[i]['tabular'] for i in range(min(50, len(dataset)))])
            samples_img = torch.stack([dataset[i]['image'] for i in range(min(50, len(dataset)))])
            explain_fn = lambda x: explainer.explain(x, samples_img[:x.shape[0]].to(device))
            df_pert = generate_perturbation_results(xm_cbm, explain_fn, samples_tab, device)
            print_results_table(df_pert, "Perturbation Robustness Results")

    # Save combined results
    combined = {'evaluation': df_eval.to_dict(orient='records')}
    with open(os.path.join(args.output_dir, 'full_results.json'), 'w') as f:
        json.dump(combined, f, indent=2, default=str)

    print(f"\nAll results saved to {args.output_dir}/")


if __name__ == '__main__':
    main()
