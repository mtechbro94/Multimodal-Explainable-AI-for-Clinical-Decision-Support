import os
import glob
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
from torchvision import transforms
from PIL import Image

class RealMIMICDataset(Dataset):
    """
    Production-grade PyTorch Dataset for Real MIMIC-IV (EHR) fused with MIMIC-CXR (Chest Radiographs).
    
    Data Schema:
    - Tabular EHR: 22 clinical features including 6 vitals (HR, SBP, DBP, SpO2, Temp, RR),
      7 laboratory markers (WBC, Hgb, Platelets, Creatinine, Bilirubin, Lactate, PaO2/FiO2),
      demographics (age, gender, ethnicity), and calculated acuity indices (SOFA, APACHE-II).
    - Medical Imaging: Paired Frontal Chest Radiographs (AP/PA views) matched within 24h
      of ICU admission, resized and normalized to (3, 224, 224).
    - Outcome: Binary in-hospital mortality (hospital_expire_flag).
    """
    
    FEATURE_NAMES = [
        'age', 'gender_female', 'ethnicity_white', 'ethnicity_black', 'ethnicity_hispanic',
        'ethnicity_asian', 'ethnicity_other',
        'heart_rate', 'sbp', 'dbp', 'spo2', 'temperature', 'respiratory_rate',
        'wbc', 'hemoglobin', 'platelets', 'creatinine', 'bilirubin', 'lactate', 'pao2_fio2_ratio',
        'sofa_score', 'apache_score'
    ]
    
    def __init__(self, cohort_csv_path, cxr_image_dir, split='train', 
                 train_val_test_ratio=(0.7, 0.15, 0.15), transform=None, seed=42):
        super().__init__()
        self.cohort_csv_path = cohort_csv_path
        self.cxr_image_dir = cxr_image_dir
        self.split = split
        self.seed = seed
        
        if not os.path.exists(cohort_csv_path):
            raise FileNotFoundError(f"Cohort metadata CSV not found at: {cohort_csv_path}")
            
        # Load matched cohort table
        self.df = pd.read_csv(cohort_csv_path)
        
        # Train / Val / Test splitting by subject_id to prevent patient data leakage across folds
        np.random.seed(seed)
        unique_subjects = self.df['subject_id'].unique()
        np.random.shuffle(unique_subjects)
        
        n_total = len(unique_subjects)
        n_train = int(n_total * train_val_test_ratio[0])
        n_val = int(n_total * train_val_test_ratio[1])
        
        train_subs = set(unique_subjects[:n_train])
        val_subs = set(unique_subjects[n_train:n_train + n_val])
        test_subs = set(unique_subjects[n_train + n_val:])
        
        if split == 'train':
            self.df = self.df[self.df['subject_id'].isin(train_subs)].reset_index(drop=True)
        elif split == 'val':
            self.df = self.df[self.df['subject_id'].isin(val_subs)].reset_index(drop=True)
        elif split == 'test':
            self.df = self.df[self.df['subject_id'].isin(test_subs)].reset_index(drop=True)
            
        # Compute/Apply normalization for tabular features
        self.tabular_features = self._prepare_tabular_features()
        self.labels = self.df['hospital_expire_flag'].values.astype(np.float32)
        
        # Imaging transforms
        if transform is not None:
            self.transform = transform
        else:
            if split == 'train':
                self.transform = transforms.Compose([
                    transforms.Resize((224, 224)),
                    transforms.RandomHorizontalFlip(p=0.5),
                    transforms.RandomRotation(degrees=7),
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
        """Standardize and impute missing tabular lab and vital measurements."""
        X = pd.DataFrame(index=self.df.index)
        
        # Demographics
        X['age'] = self.df['age'].fillna(self.df['age'].median())
        X['gender_female'] = (self.df['gender'].str.upper() == 'F').astype(float)
        
        # One-hot ethnicity
        eth = self.df['ethnicity'].str.upper().fillna('OTHER')
        X['ethnicity_white'] = eth.str.contains('WHITE').astype(float)
        X['ethnicity_black'] = eth.str.contains('BLACK').astype(float)
        X['ethnicity_hispanic'] = eth.str.contains('HISPANIC').astype(float)
        X['ethnicity_asian'] = eth.str.contains('ASIAN').astype(float)
        X['ethnicity_other'] = (~(X['ethnicity_white'] | X['ethnicity_black'] | 
                                  X['ethnicity_hispanic'] | X['ethnicity_asian'])).astype(float)
        
        # Continuous Vitals & Labs with standard median imputation
        continuous_cols = [
            'heart_rate', 'sbp', 'dbp', 'spo2', 'temperature', 'respiratory_rate',
            'wbc', 'hemoglobin', 'platelets', 'creatinine', 'bilirubin', 'lactate', 
            'pao2_fio2_ratio', 'sofa_score', 'apache_score'
        ]
        
        for col in continuous_cols:
            if col in self.df.columns:
                val = self.df[col]
                med = val.median() if not pd.isna(val.median()) else 0.0
                std = val.std() if (val.std() is not None and val.std() > 1e-5) else 1.0
                X[col] = ((val.fillna(med) - med) / std).astype(float)
            else:
                X[col] = 0.0
                
        return X[self.FEATURE_NAMES].values.astype(np.float32)

    def _resolve_image_path(self, row):
        """Locates the JPG chest radiograph from study_id and dicom_id."""
        if 'image_path' in row and pd.notna(row['image_path']) and os.path.exists(str(row['image_path'])):
            return str(row['image_path'])
            
        subject_id = str(int(row['subject_id']))
        study_id = str(int(row['study_id']))
        dicom_id = str(row['dicom_id'])
        
        # Standard MIMIC-CXR-JPG folder structure: pXX/pXXXXXXXX/sXXXXXXXX/XXXXXXXX.jpg
        prefix = f"p{subject_id[:2]}"
        candidate_path = os.path.join(self.cxr_image_dir, prefix, f"p{subject_id}", f"s{study_id}", f"{dicom_id}.jpg")
        if os.path.exists(candidate_path):
            return candidate_path
            
        # Fallback shallow directory check
        candidate_shallow = os.path.join(self.cxr_image_dir, f"{dicom_id}.jpg")
        if os.path.exists(candidate_shallow):
            return candidate_shallow
            
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
            'label': label,
            'subject_id': int(row['subject_id']),
            'stay_id': int(row.get('stay_id', -1))
        }

    @classmethod
    def get_feature_names(cls):
        return cls.FEATURE_NAMES

    @classmethod
    def get_tabular_dim(cls):
        return len(cls.FEATURE_NAMES)
