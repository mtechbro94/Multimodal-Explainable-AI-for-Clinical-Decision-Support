import argparse
import os
import sys
import json
import random
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
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

def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def get_device():
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")

def collate_fn(batch):
    tabular = torch.stack([item['tabular'] for item in batch])
    image = torch.stack([item['image'] for item in batch])
    label = torch.stack([item['label'] for item in batch])
    return {'tabular': tabular, 'image': image, 'label': label}

def train_tree_model(model_class, X_train, y_train, X_val, y_val):
    model = model_class()
    model.fit(X_train, y_train, eval_set=[(X_val, y_val)])
    y_pred = model.predict_proba(X_val)
    val_auroc = roc_auc_score(y_val, y_pred)
    return model, {'val_auroc': val_auroc}

def train_neural_epoch(model, loader, optimizer, criterion, device, is_multimodal=False, loss_fn=None):
    model.train()
    total_loss = 0.0
    for batch in loader:
        optimizer.zero_grad()
        targets = batch['label'].to(device)
        
        if is_multimodal:
            x_tab = batch['tabular'].to(device)
            x_img = batch['image'].to(device)
            
            if isinstance(model, CrossModalConceptBottleneck):
                logits, concept_dict = model(x_tab, x_img)
                loss, loss_comp = loss_fn(logits, targets, concept_dict, model, x_tab, x_img)
            else:
                logits = model(x_tab, x_img)
                loss = criterion(logits, targets)
        else:
            if hasattr(model, 'get_cam_target_layer') or hasattr(model, 'get_attention_weights'): # Image
                x = batch['image'].to(device)
            else:
                x = batch['tabular'].to(device)
            logits = model(x)
            loss = criterion(logits, targets)
            
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        total_loss += loss.item()
        
    return total_loss / len(loader), None

def validate_neural(model, loader, device, is_multimodal=False):
    model.eval()
    y_true = []
    y_prob = []
    with torch.no_grad():
        for batch in loader:
            targets = batch['label']
            if is_multimodal:
                x_tab = batch['tabular'].to(device)
                x_img = batch['image'].to(device)
                if isinstance(model, CrossModalConceptBottleneck):
                    logits, _ = model(x_tab, x_img)
                else:
                    logits = model(x_tab, x_img)
            else:
                if hasattr(model, 'get_cam_target_layer') or hasattr(model, 'get_attention_weights'):
                    x = batch['image'].to(device)
                else:
                    x = batch['tabular'].to(device)
                logits = model(x)
                
            probs = torch.sigmoid(logits).cpu()
            y_true.extend(targets.numpy())
            y_prob.extend(probs.numpy())
            
    return np.array(y_true), np.array(y_prob)

def train_neural_model(model, train_loader, val_loader, epochs, lr, device, model_name, output_dir, patience=7, loss_fn=None):
    optimizer = optim.AdamW(model.parameters(), lr=lr)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    if loss_fn is None:
        criterion = nn.BCEWithLogitsLoss()
    else:
        criterion = loss_fn
        
    is_multimodal = model_name in ['late_fusion', 'xm_cbm']
    history = {'train_loss': [], 'val_auroc': []}
    best_auroc = -1
    patience_counter = 0
    os.makedirs(os.path.join(output_dir, 'checkpoints'), exist_ok=True)
    
    for ep in tqdm(range(epochs), desc=f'Training {model_name}'):
        train_loss, _ = train_neural_epoch(model, train_loader, optimizer, criterion, device, is_multimodal, loss_fn if isinstance(model, CrossModalConceptBottleneck) else None)
        scheduler.step()
        
        y_true, y_prob = validate_neural(model, val_loader, device, is_multimodal)
        val_auroc = roc_auc_score(y_true, y_prob)
        
        history['train_loss'].append(train_loss)
        history['val_auroc'].append(val_auroc)
        
        if val_auroc > best_auroc:
            best_auroc = val_auroc
            torch.save(model.state_dict(), os.path.join(output_dir, 'checkpoints', f'{model_name}_best.pt'))
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= patience:
                break
                
    model.load_state_dict(torch.load(os.path.join(output_dir, 'checkpoints', f'{model_name}_best.pt')))
    return history

def run_single_fold(fold_idx, train_indices, val_indices, dataset, config, device, output_dir):
    train_loader = DataLoader(Subset(dataset, train_indices), batch_size=config.batch_size, shuffle=True, collate_fn=collate_fn)
    val_loader = DataLoader(Subset(dataset, val_indices), batch_size=config.batch_size, shuffle=False, collate_fn=collate_fn)
    
    model_name = config.model
    tab_dim = dataset.get_tabular_dim()
    
    if model_name in ['xgboost', 'lightgbm']:
        X_train = np.stack([dataset[i]['tabular'].numpy() for i in train_indices])
        y_train = np.array([dataset[i]['label'].numpy() for i in train_indices])
        X_val = np.stack([dataset[i]['tabular'].numpy() for i in val_indices])
        y_val = np.array([dataset[i]['label'].numpy() for i in val_indices])
        
        model_class = XGBoostWrapper if model_name == 'xgboost' else LightGBMWrapper
        model, metrics = train_tree_model(model_class, X_train, y_train, X_val, y_val)
        y_prob = model.predict_proba(X_val)
        return y_val, y_prob, model
        
    if model_name == 'mlp':
        model = TabularMLP(input_dim=tab_dim).to(device)
    elif model_name == 'densenet':
        model = DenseNet121Classifier().to(device)
    elif model_name == 'vit':
        model = ViTClassifier().to(device)
    elif model_name == 'late_fusion':
        model = LateFusionModel(tab_dim=tab_dim).to(device)
    elif model_name == 'xm_cbm':
        model = CrossModalConceptBottleneck(tab_dim=tab_dim).to(device)
        
    loss_fn = XMCBMLoss() if model_name == 'xm_cbm' else None
    
    history = train_neural_model(model, train_loader, val_loader, config.epochs, config.lr, device, f"{model_name}_fold{fold_idx}", output_dir, loss_fn=loss_fn)
    y_true, y_prob = validate_neural(model, val_loader, device, model_name in ['late_fusion', 'xm_cbm'])
    
    return y_true, y_prob, model

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, default='all')
    parser.add_argument('--n_samples', type=int, default=3000)
    parser.add_argument('--epochs', type=int, default=30)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--n_folds', type=int, default=5)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--output_dir', type=str, default='../results')
    args = parser.parse_args()
    
    set_seed(args.seed)
    device = get_device()
    os.makedirs(args.output_dir, exist_ok=True)
    
    dataset = SyntheticMIMICDataset(n_samples=args.n_samples)
    labels = [dataset[i]['label'].item() for i in range(len(dataset))]
    skf = StratifiedKFold(n_splits=args.n_folds, shuffle=True, random_state=args.seed)
    
    models_to_train = ['xgboost', 'lightgbm', 'mlp', 'densenet', 'vit', 'late_fusion', 'xm_cbm'] if args.model == 'all' else [args.model]
    
    results = {}
    for model_name in models_to_train:
        args.model = model_name
        aucs = []
        for fold, (train_idx, val_idx) in enumerate(skf.split(np.zeros(len(labels)), labels)):
            y_true, y_prob, model = run_single_fold(fold, train_idx, val_idx, dataset, args, device, args.output_dir)
            auc = roc_auc_score(y_true, y_prob)
            aucs.append(auc)
        results[model_name] = {'mean_auc': np.mean(aucs), 'std_auc': np.std(aucs)}
        
    with open(os.path.join(args.output_dir, 'training_summary.json'), 'w') as f:
        json.dump(results, f, indent=4)
        
    print("Training Summary:")
    for m, r in results.items():
        print(f"{m}: AUC = {r['mean_auc']:.4f} ± {r['std_auc']:.4f}")

if __name__ == '__main__':
    main()
