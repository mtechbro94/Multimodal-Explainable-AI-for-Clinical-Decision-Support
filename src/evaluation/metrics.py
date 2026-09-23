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

def compute_confusion_matrix(y_true, y_pred):
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    tn = int(np.sum((y_true == 0) & (y_pred == 0)))
    fp = int(np.sum((y_true == 0) & (y_pred == 1)))
    fn = int(np.sum((y_true == 1) & (y_pred == 0)))
    tp = int(np.sum((y_true == 1) & (y_pred == 1)))
    return {"TN": tn, "FP": fp, "FN": fn, "TP": tp}

def compute_accuracy(y_true, y_pred):
    cm = compute_confusion_matrix(y_true, y_pred)
    total = cm["TP"] + cm["TN"] + cm["FP"] + cm["FN"]
    return (cm["TP"] + cm["TN"]) / total if total > 0 else 0.0

def compute_sensitivity(y_true, y_pred):
    """Sensitivity / Recall / True Positive Rate (TPR)"""
    cm = compute_confusion_matrix(y_true, y_pred)
    positives = cm["TP"] + cm["FN"]
    return cm["TP"] / positives if positives > 0 else 0.0

def compute_specificity(y_true, y_pred):
    """Specificity / True Negative Rate (TNR)"""
    cm = compute_confusion_matrix(y_true, y_pred)
    negatives = cm["TN"] + cm["FP"]
    return cm["TN"] / negatives if negatives > 0 else 0.0

def compute_precision(y_true, y_pred):
    """Positive Predictive Value (PPV)"""
    cm = compute_confusion_matrix(y_true, y_pred)
    pred_positives = cm["TP"] + cm["FP"]
    return cm["TP"] / pred_positives if pred_positives > 0 else 0.0

def compute_npv(y_true, y_pred):
    """Negative Predictive Value (NPV)"""
    cm = compute_confusion_matrix(y_true, y_pred)
    pred_negatives = cm["TN"] + cm["FN"]
    return cm["TN"] / pred_negatives if pred_negatives > 0 else 0.0

def compute_f1(y_true, y_pred):
    prec = compute_precision(y_true, y_pred)
    rec = compute_sensitivity(y_true, y_pred)
    return 2 * (prec * rec) / (prec + rec) if (prec + rec) > 0 else 0.0

def compute_balanced_accuracy(y_true, y_pred):
    sens = compute_sensitivity(y_true, y_pred)
    spec = compute_specificity(y_true, y_pred)
    return (sens + spec) / 2.0

def compute_mcc(y_true, y_pred):
    """Matthews Correlation Coefficient (MCC)"""
    cm = compute_confusion_matrix(y_true, y_pred)
    tp, tn, fp, fn = cm["TP"], cm["TN"], cm["FP"], cm["FN"]
    numerator = (tp * tn) - (fp * fn)
    denominator = np.sqrt(float((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn)))
    return numerator / denominator if denominator > 0 else 0.0

def find_optimal_threshold_youden(y_true, y_prob):
    """Determine threshold that maximizes Youden's J statistic (Sensitivity + Specificity - 1)."""
    thresholds = np.linspace(0.05, 0.95, 100)
    best_j = -1.0
    best_thresh = 0.5
    for t in thresholds:
        pred = (y_prob >= t).astype(int)
        j_val = compute_sensitivity(y_true, pred) + compute_specificity(y_true, pred) - 1.0
        if j_val > best_j:
            best_j = j_val
            best_thresh = t
    return best_thresh

def compute_all_metrics(y_true, y_prob, threshold=None):
    y_true = np.asarray(y_true)
    y_prob = np.asarray(y_prob)
    
    if threshold is None:
        threshold = find_optimal_threshold_youden(y_true, y_prob)
        
    y_pred = (y_prob >= threshold).astype(int)
    cm = compute_confusion_matrix(y_true, y_pred)
    
    return {
        'auroc': compute_auroc(y_true, y_prob),
        'auprc': compute_auprc(y_true, y_prob),
        'brier': compute_brier_score(y_true, y_prob),
        'ece': compute_ece(y_true, y_prob),
        'threshold': float(threshold),
        'accuracy': compute_accuracy(y_true, y_pred),
        'balanced_accuracy': compute_balanced_accuracy(y_true, y_pred),
        'sensitivity': compute_sensitivity(y_true, y_pred),
        'recall': compute_sensitivity(y_true, y_pred),
        'specificity': compute_specificity(y_true, y_pred),
        'precision': compute_precision(y_true, y_pred),
        'npv': compute_npv(y_true, y_pred),
        'f1': compute_f1(y_true, y_pred),
        'mcc': compute_mcc(y_true, y_pred),
        'confusion_matrix': cm
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
