import torch
import numpy as np
from sklearn.metrics import auc

def _to_tensor(x):
    if isinstance(x, np.ndarray):
        return torch.from_numpy(x).float()
    return x.clone().detach().float()

def _to_numpy(x):
    if torch.is_tensor(x):
        return x.detach().cpu().numpy()
    return np.array(x)

def compute_deletion_auc(model_fn, x, attributions, n_steps=20):
    x = _to_tensor(x)
    attributions = _to_numpy(attributions)
    
    is_image = x.dim() > 2
    
    # Flatten features for ranking
    if is_image:
        flat_attrs = attributions.reshape(-1)
        # Assuming channel dim is 1 for images (B, C, H, W), rank pixels (H, W)
        spatial_attrs = np.abs(attributions).mean(axis=(0, 1)) # HW
        flat_attrs = spatial_attrs.flatten()
    else:
        flat_attrs = np.abs(attributions).flatten()
        
    sort_indices = np.argsort(-flat_attrs) # Descending
    
    predictions = []
    
    x_masked = x.clone()
    
    step_size = len(flat_attrs) // n_steps
    
    for i in range(n_steps + 1):
        num_mask = min(i * step_size, len(flat_attrs))
        mask_indices = sort_indices[:num_mask]
        
        x_curr = x_masked.clone()
        
        if is_image:
            H, W = x.shape[2:]
            for idx in mask_indices:
                r, c = divmod(idx, W)
                x_curr[:, :, r, c] = x.mean(dim=(2,3), keepdim=True) # Mean pixel masking
        else:
            for idx in mask_indices:
                x_curr[:, idx] = 0.0 # Zero masking
                
        with torch.no_grad():
            pred = model_fn(x_curr)
            if torch.is_tensor(pred):
                pred = pred.softmax(dim=1).max().item()
            predictions.append(pred)
            
    fractions = np.linspace(0, 1, n_steps + 1)
    return auc(fractions, predictions)

def compute_insertion_auc(model_fn, x, attributions, n_steps=20):
    x = _to_tensor(x)
    attributions = _to_numpy(attributions)
    
    is_image = x.dim() > 2
    
    if is_image:
        spatial_attrs = np.abs(attributions).mean(axis=(0, 1))
        flat_attrs = spatial_attrs.flatten()
        baseline = x.mean(dim=(2,3), keepdim=True).expand_as(x).clone()
    else:
        flat_attrs = np.abs(attributions).flatten()
        baseline = torch.zeros_like(x)
        
    sort_indices = np.argsort(-flat_attrs)
    
    predictions = []
    step_size = len(flat_attrs) // n_steps
    
    for i in range(n_steps + 1):
        num_insert = min(i * step_size, len(flat_attrs))
        insert_indices = sort_indices[:num_insert]
        
        x_curr = baseline.clone()
        
        if is_image:
            H, W = x.shape[2:]
            for idx in insert_indices:
                r, c = divmod(idx, W)
                x_curr[:, :, r, c] = x[:, :, r, c]
        else:
            for idx in insert_indices:
                x_curr[:, idx] = x[:, idx]
                
        with torch.no_grad():
            pred = model_fn(x_curr)
            if torch.is_tensor(pred):
                pred = pred.softmax(dim=1).max().item()
            predictions.append(pred)
            
    fractions = np.linspace(0, 1, n_steps + 1)
    return auc(fractions, predictions)

def compute_infidelity(model_fn, x, attributions, n_perturbations=100, sigma=0.1):
    x = _to_tensor(x)
    attributions = _to_tensor(attributions)
    
    with torch.no_grad():
        f_x = model_fn(x)
        
    infidelities = []
    
    for _ in range(n_perturbations):
        mu = torch.randn_like(x) * sigma
        x_pert = x - mu
        
        with torch.no_grad():
            f_x_pert = model_fn(x_pert)
            
        pred_diff = (f_x - f_x_pert).view(-1)
        
        dot_product = torch.sum(mu.flatten() * attributions.flatten())
        
        infidelity = ((dot_product - pred_diff) ** 2).mean().item()
        infidelities.append(infidelity)
        
    return np.mean(infidelities)

def compute_all_faithfulness(model_fn, x, attributions, n_steps=20, n_perturbations=100):
    return {
        'deletion_auc': compute_deletion_auc(model_fn, x, attributions, n_steps),
        'insertion_auc': compute_insertion_auc(model_fn, x, attributions, n_steps),
        'infidelity': compute_infidelity(model_fn, x, attributions, n_perturbations)
    }
