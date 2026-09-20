import torch
import numpy as np
from scipy.stats import spearmanr

def _to_tensor(x):
    if isinstance(x, np.ndarray):
        return torch.from_numpy(x).float()
    return x.clone().detach().float()

def estimate_lipschitz_constant(explain_fn, x, n_perturbations=50, epsilon=0.01):
    x = _to_tensor(x)
    
    orig_explanation = _to_tensor(explain_fn(x)).flatten()
    
    lipschitz_constants = []
    
    for _ in range(n_perturbations):
        noise = torch.randn_like(x)
        noise = noise / torch.norm(noise) * epsilon
        x_pert = x + noise
        
        pert_explanation = _to_tensor(explain_fn(x_pert)).flatten()
        
        exp_diff = torch.norm(orig_explanation - pert_explanation)
        x_diff = torch.norm(x - x_pert)
        
        lipschitz_constants.append((exp_diff / (x_diff + 1e-8)).item())
        
    return {
        'max': np.max(lipschitz_constants),
        'mean': np.mean(lipschitz_constants)
    }

def explanation_sensitivity(explain_fn, x, sigmas=[0.01, 0.05, 0.1]):
    x = _to_tensor(x)
    orig_explanation = _to_tensor(explain_fn(x)).flatten().numpy()
    
    sensitivities = {}
    
    for sigma in sigmas:
        noise = torch.randn_like(x) * sigma
        x_pert = x + noise
        
        pert_explanation = _to_tensor(explain_fn(x_pert)).flatten().numpy()
        
        correlation, _ = spearmanr(orig_explanation, pert_explanation)
        sensitivities[sigma] = correlation
        
    return sensitivities

def topk_stability(explain_fn, x, k=5, n_perturbations=20, epsilon=0.01):
    x = _to_tensor(x)
    orig_explanation = _to_tensor(explain_fn(x)).flatten()
    
    _, orig_topk = torch.topk(torch.abs(orig_explanation), min(k, len(orig_explanation)))
    orig_topk_set = set(orig_topk.tolist())
    
    jaccards = []
    
    for _ in range(n_perturbations):
        noise = torch.randn_like(x)
        noise = noise / torch.norm(noise) * epsilon
        x_pert = x + noise
        
        pert_explanation = _to_tensor(explain_fn(x_pert)).flatten()
        _, pert_topk = torch.topk(torch.abs(pert_explanation), min(k, len(pert_explanation)))
        pert_topk_set = set(pert_topk.tolist())
        
        intersection = len(orig_topk_set.intersection(pert_topk_set))
        union = len(orig_topk_set.union(pert_topk_set))
        
        jaccard = intersection / union if union > 0 else 0
        jaccards.append(jaccard)
        
    return np.mean(jaccards)
