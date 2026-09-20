import numpy as np
from sklearn.metrics import roc_auc_score, average_precision_score, brier_score_loss

def compute_auroc(y_true, y_prob):
    try:
        return roc_auc_score(y_true, y_prob)
    except ValueError:
        return np.nan

def compute_auprc(y_true, y_prob):
    try:
        return average_precision_score(y_true, y_prob)
    except ValueError:
        return np.nan
        
def compute_brier_score(y_true, y_prob):
    try:
        return brier_score_loss(y_true, y_prob)
    except ValueError:
        return np.nan

def compute_ece(y_true, y_prob, n_bins=15):
    bin_boundaries = np.linspace(0, 1, n_bins + 1)
    bin_lowers = bin_boundaries[:-1]
    bin_uppers = bin_boundaries[1:]
    
    ece = 0.0
    for bin_lower, bin_upper in zip(bin_lowers, bin_uppers):
        in_bin = (y_prob > bin_lower) & (y_prob <= bin_upper)
        prop_in_bin = in_bin.mean()
        
        if prop_in_bin > 0:
            accuracy_in_bin = y_true[in_bin].mean()
            avg_confidence_in_bin = y_prob[in_bin].mean()
            ece += np.abs(avg_confidence_in_bin - accuracy_in_bin) * prop_in_bin
            
    return ece

def compute_all_metrics(y_true, y_prob):
    return {
        'auroc': compute_auroc(y_true, y_prob),
        'auprc': compute_auprc(y_true, y_prob),
        'brier': compute_brier_score(y_true, y_prob),
        'ece': compute_ece(y_true, y_prob)
    }

def compute_metrics_with_ci(y_true, y_prob, n_bootstrap=1000, ci=0.95):
    n = len(y_true)
    bootstrap_metrics = {'auroc': [], 'auprc': [], 'brier': [], 'ece': []}
    
    for _ in range(n_bootstrap):
        indices = np.random.choice(n, n, replace=True)
        y_true_boot = y_true[indices]
        y_prob_boot = y_prob[indices]
        
        if len(np.unique(y_true_boot)) < 2:
            continue
            
        metrics = compute_all_metrics(y_true_boot, y_prob_boot)
        for k, v in metrics.items():
            bootstrap_metrics[k].append(v)
            
    results = {}
    lower_percentile = (1 - ci) / 2 * 100
    upper_percentile = (1 + ci) / 2 * 100
    
    for k, v in bootstrap_metrics.items():
        if len(v) > 0:
            results[k] = {
                'mean': np.mean(v),
                'lower': np.percentile(v, lower_percentile),
                'upper': np.percentile(v, upper_percentile)
            }
            
    return results
