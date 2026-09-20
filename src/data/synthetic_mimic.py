import numpy as np
import torch
from torch.utils.data import Dataset
from scipy.ndimage import gaussian_filter

class SyntheticMIMICDataset(Dataset):
    """
    Synthetic dataset mimicking MIMIC-IV tabular data and MIMIC-CXR images.
    """
    def __init__(self, num_samples=3000, split='train', seed=42):
        super().__init__()
        self.num_samples = num_samples
        self.split = split
        self.seed = seed
        
        # Split configurations
        if split == 'train':
            split_seed = seed
        elif split == 'val':
            split_seed = seed + 1
        elif split == 'test':
            split_seed = seed + 2
        else:
            raise ValueError(f"Unknown split {split}")
            
        np.random.seed(split_seed)
        torch.manual_seed(split_seed)
        
        self._generate_data()
        
    def _generate_data(self):
        N = self.num_samples
        
        # 1. Demographics
        age = np.clip(np.random.normal(65, 15, N), 18, 100)
        sex = np.random.binomial(1, 0.5, N)  # 0 or 1
        ethnicity = np.random.multinomial(1, [0.6, 0.15, 0.1, 0.1, 0.05], N) # 5 categories
        
        # 2. Vitals
        heart_rate = np.random.normal(85, 18, N)
        sbp = np.random.normal(125, 22, N)
        dbp = np.random.normal(75, 14, N)
        spo2 = np.clip(np.random.normal(95, 4, N), 60, 100)
        temp = np.random.normal(37.0, 0.6, N)
        resp_rate = np.random.normal(18, 5, N)
        
        # 3. Labs
        wbc = np.random.normal(10.5, 5.2, N)
        hemo = np.random.normal(12.0, 2.5, N)
        platelets = np.random.normal(220, 90, N)
        creat = np.random.normal(1.2, 0.9, N)
        bili = np.random.normal(1.5, 2.0, N)
        lactate = np.random.normal(2.0, 1.8, N)
        pao2_fio2 = np.random.normal(280, 100, N)
        
        # 4. Derived Scores (Simplified)
        sofa = (creat > 1.2).astype(float) + (bili > 1.2).astype(float) + (platelets < 150).astype(float) + (sbp < 100).astype(float)
        apache = (age > 65).astype(float) + (heart_rate > 100).astype(float) + (resp_rate > 25).astype(float)
        
        # Combine into tabular array
        self.tabular_data = np.column_stack((
            age, sex, ethnicity, 
            heart_rate, sbp, dbp, spo2, temp, resp_rate,
            wbc, hemo, platelets, creat, bili, lactate, pao2_fio2,
            sofa, apache
        ))
        
        # Feature names mapping
        self.feature_names = [
            'age', 'sex', 
            'eth_1', 'eth_2', 'eth_3', 'eth_4', 'eth_5',
            'heart_rate', 'sbp', 'dbp', 'spo2', 'temp', 'resp_rate',
            'wbc', 'hemo', 'platelets', 'creat', 'bili', 'lactate', 'pao2_fio2',
            'sofa', 'apache'
        ]
        
        # Generate Mortality Label via logistic function
        # Risk factors: high lactate, low pao2_fio2, high creatinine, old age, low sbp
        risk_score = (
            0.5 * (lactate - 2.0)/1.8 - 
            0.4 * (pao2_fio2 - 280)/100 + 
            0.3 * (creat - 1.2)/0.9 + 
            0.4 * (age - 65)/15 - 
            0.5 * (sbp - 125)/22 - 
            1.5 # Bias to shift prevalence to ~15%
        )
        prob = 1 / (1 + np.exp(-risk_score))
        self.labels = np.random.binomial(1, prob)
        
        # Generate paired chest radiographs
        self.images = []
        for i in range(N):
            img = self._generate_single_image(self.labels[i])
            self.images.append(img)
            
        self.tabular_data = torch.tensor(self.tabular_data, dtype=torch.float32)
        self.labels = torch.tensor(self.labels, dtype=torch.float32)
        
    def _generate_single_image(self, is_positive):
        # Base perlin noise-like background
        noise = np.random.uniform(0, 1, (224, 224))
        base_img = gaussian_filter(noise, sigma=3)
        base_img = (base_img - base_img.min()) / (base_img.max() - base_img.min() + 1e-8)
        
        # Embed opacity if positive
        if is_positive:
            patch = np.zeros((224, 224))
            cy, cx = np.random.randint(50, 174, 2)
            patch[cy, cx] = 1.0
            opacity = gaussian_filter(patch, sigma=15)
            opacity = opacity / (opacity.max() + 1e-8)
            intensity = np.random.uniform(0.5, 1.0)
            base_img = base_img + opacity * intensity
            base_img = np.clip(base_img, 0, 1)
            
        # 3 Channels
        img_3c = np.stack((base_img,)*3, axis=-1)
        
        # Normalize to ImageNet stats
        mean = np.array([0.485, 0.456, 0.406])
        std = np.array([0.229, 0.224, 0.225])
        img_norm = (img_3c - mean) / std
        
        # HWC to CHW
        img_chw = img_norm.transpose((2, 0, 1))
        return torch.tensor(img_chw, dtype=torch.float32)

    def get_feature_names(self):
        return self.feature_names
        
    def get_tabular_dim(self):
        return len(self.feature_names)

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        return {
            'tabular': self.tabular_data[idx],
            'image': self.images[idx],
            'label': self.labels[idx]
        }
