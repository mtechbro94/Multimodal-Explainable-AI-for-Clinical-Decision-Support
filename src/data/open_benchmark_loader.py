import os
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
from torchvision import transforms
from PIL import Image

class OpenClinicalBenchmarkDataset(Dataset):
    """
    Open-Access Real Clinical Multimodal Benchmark Dataset.
    Designed for immediate execution on Google Colab or Kaggle without credential delays.
    
    Supports:
    1. Stanford CheXpert or NIH ChestX-ray14 imaging data (real patient frontal chest radiographs).
    2. Real clinical metadata: patient demographics, age, sex, AP/PA projection,
       comorbidities, and multi-label pathology (Pneumonia, Cardiomegaly, Atelectasis, Edema, Effusion).
    3. Binary triage / acute acuity prediction task (e.g. Critical Acuity / High-Risk Cardiopulmonary Pathology).
    """
    
    FEATURE_NAMES = [
        'age_normalized', 'sex_male', 'view_ap',
        'enlarged_cardiomediastinum', 'cardiomegaly', 'lung_opacity',
        'lung_lesion', 'edema', 'consolidation', 'pneumonia',
        'atelectasis', 'pneumothorax', 'pleural_effusion', 'pleural_other',
        'fracture', 'support_devices'
    ]
    
    def __init__(self, metadata_csv_path, image_root_dir, split='train',
                 train_val_test_ratio=(0.7, 0.15, 0.15), transform=None, seed=42):
        super().__init__()
        self.metadata_csv_path = metadata_csv_path
        self.image_root_dir = image_root_dir
        self.split = split
        
        if not os.path.exists(metadata_csv_path):
            raise FileNotFoundError(f"Open benchmark metadata CSV not found: {metadata_csv_path}")
            
        self.df = pd.read_csv(metadata_csv_path)
        
        # Patient-level splitting (using 'Path' patient identifier or 'Patient ID')
        patient_col = 'Patient ID' if 'Patient ID' in self.df.columns else 'subject_id'
        if patient_col not in self.df.columns and 'Path' in self.df.columns:
            self.df['patient_id'] = self.df['Path'].apply(lambda x: x.split('/')[2] if len(x.split('/')) > 2 else x)
            patient_col = 'patient_id'
        elif patient_col not in self.df.columns:
            self.df['patient_id'] = np.arange(len(self.df))
            patient_col = 'patient_id'
            
        np.random.seed(seed)
        unique_pts = self.df[patient_col].unique()
        np.random.shuffle(unique_pts)
        
        n_total = len(unique_pts)
        n_train = int(n_total * train_val_test_ratio[0])
        n_val = int(n_total * train_val_test_ratio[1])
        
        train_pts = set(unique_pts[:n_train])
        val_pts = set(unique_pts[n_train:n_train + n_val])
        test_pts = set(unique_pts[n_train + n_val:])
        
        if split == 'train':
            self.df = self.df[self.df[patient_col].isin(train_pts)].reset_index(drop=True)
        elif split == 'val':
            self.df = self.df[self.df[patient_col].isin(val_pts)].reset_index(drop=True)
        elif split == 'test':
            self.df = self.df[self.df[patient_col].isin(test_pts)].reset_index(drop=True)
            
        self.tabular_features = self._prepare_tabular_features()
        
        # Target label: High-acuity critical finding (Pneumonia, Edema, or Consolidation present)
        # or explicit target column if present
        if 'mortality' in self.df.columns:
            self.labels = self.df['mortality'].values.astype(np.float32)
        elif 'critical_finding' in self.df.columns:
            self.labels = self.df['critical_finding'].values.astype(np.float32)
        else:
            # Derive composite critical triage target from pathology annotations
            crit_cols = [c for c in ['Pneumonia', 'Edema', 'Consolidation'] if c in self.df.columns]
            if crit_cols:
                # 1 if any critical pathology is positively present (1.0), else 0
                self.labels = (self.df[crit_cols] == 1.0).any(axis=1).values.astype(np.float32)
            else:
                self.labels = np.zeros(len(self.df), dtype=np.float32)
                
        # Image transforms
        if transform is not None:
            self.transform = transform
        else:
            if split == 'train':
                self.transform = transforms.Compose([
                    transforms.Resize((224, 224)),
                    transforms.RandomHorizontalFlip(p=0.5),
                    transforms.ToTensor(),
                    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
                ])
            else:
                self.transform = transforms.Compose([
                    transforms.Resize((224, 224)),
                    transforms.ToTensor(),
                    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
                ])

    def _prepare_tabular_features(self):
        X = pd.DataFrame(index=self.df.index)
        
        # Age
        age_col = 'Age' if 'Age' in self.df.columns else 'age'
        if age_col in self.df.columns:
            age = pd.to_numeric(self.df[age_col], errors='coerce').fillna(60.0)
            X['age_normalized'] = (age - 60.0) / 18.0
        else:
            X['age_normalized'] = 0.0
            
        # Sex
        sex_col = 'Sex' if 'Sex' in self.df.columns else 'gender'
        if sex_col in self.df.columns:
            X['sex_male'] = (self.df[sex_col].astype(str).str.upper() == 'M').astype(float)
        else:
            X['sex_male'] = 0.5
            
        # View position
        vp_col = 'Frontal/Lateral' if 'Frontal/Lateral' in self.df.columns else 'ViewPosition'
        if vp_col in self.df.columns:
            X['view_ap'] = self.df[vp_col].astype(str).str.contains('AP').astype(float)
        else:
            X['view_ap'] = 1.0
            
        # Pathology feature columns (CheXpert 14 label standard)
        path_labels = [
            ('enlarged_cardiomediastinum', 'Enlarged Cardiomediastinum'),
            ('cardiomegaly', 'Cardiomegaly'),
            ('lung_opacity', 'Lung Opacity'),
            ('lung_lesion', 'Lung Lesion'),
            ('edema', 'Edema'),
            ('consolidation', 'Consolidation'),
            ('pneumonia', 'Pneumonia'),
            ('atelectasis', 'Atelectasis'),
            ('pneumothorax', 'Pneumothorax'),
            ('pleural_effusion', 'Pleural Effusion'),
            ('pleural_other', 'Pleural Other'),
            ('fracture', 'Fracture'),
            ('support_devices', 'Support Devices')
        ]
        
        for feat_name, col_name in path_labels:
            if col_name in self.df.columns:
                # CheXpert coding: 1=positive, 0=negative, -1=uncertain (mapped to 0.5), NaN=unmentioned (0)
                val = self.df[col_name].replace(-1.0, 0.5).fillna(0.0).astype(float)
                X[feat_name] = val
            else:
                X[feat_name] = 0.0
                
        return X[self.FEATURE_NAMES].values.astype(np.float32)

    def _resolve_image_path(self, row):
        if 'Path' in row and pd.notna(row['Path']):
            direct = str(row['Path'])
            if os.path.exists(direct):
                return direct
            joined = os.path.join(self.image_root_dir, direct)
            if os.path.exists(joined):
                return joined
            # Strip CheXpert-v1.0-small prefix if relative
            rel = direct.replace('CheXpert-v1.0-small/', '').replace('CheXpert-v1.0/', '')
            joined2 = os.path.join(self.image_root_dir, rel)
            if os.path.exists(joined2):
                return joined2
                
        if 'image_name' in row and pd.notna(row['image_name']):
            cand = os.path.join(self.image_root_dir, str(row['image_name']))
            if os.path.exists(cand):
                return cand
                
        return None

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        tab_vector = torch.from_numpy(self.tabular_features[idx]).float()
        label = torch.tensor(self.labels[idx], dtype=torch.float32)
        
        img_path = self._resolve_image_path(row)
        if img_path and os.path.exists(img_path):
            try:
                img = Image.open(img_path).convert('RGB')
                img_tensor = self.transform(img)
            except Exception:
                img_tensor = torch.zeros((3, 224, 224), dtype=torch.float32)
        else:
            img_tensor = torch.zeros((3, 224, 224), dtype=torch.float32)
            
        return {
            'tabular': tab_vector,
            'image': img_tensor,
            'label': label
        }

    @classmethod
    def get_feature_names(cls):
        return cls.FEATURE_NAMES

    @classmethod
    def get_tabular_dim(cls):
        return len(cls.FEATURE_NAMES)
