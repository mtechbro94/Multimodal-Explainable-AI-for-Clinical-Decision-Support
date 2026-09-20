import torch
import torch.nn as nn
import xgboost as xgb
import lightgbm as lgb

class TabularMLP(nn.Module):
    """
    Tabular Baseline Model: 3-layer MLP with BatchNorm, ReLU, and Dropout.
    """
    def __init__(self, input_dim):
        super().__init__()
        
        self.fc1 = nn.Linear(input_dim, 256)
        self.bn1 = nn.BatchNorm1d(256)
        
        self.fc2 = nn.Linear(256, 128)
        self.bn2 = nn.BatchNorm1d(128)
        
        self.fc3 = nn.Linear(128, 64)
        self.bn3 = nn.BatchNorm1d(64)
        
        self.dropout = nn.Dropout(0.3)
        self.relu = nn.ReLU()
        
        self.classifier = nn.Linear(64, 1)

    def get_embedding(self, x):
        """Returns the penultimate layer output."""
        x = self.relu(self.bn1(self.fc1(x)))
        x = self.dropout(x)
        x = self.relu(self.bn2(self.fc2(x)))
        x = self.dropout(x)
        x = self.relu(self.bn3(self.fc3(x)))
        return x

    def forward(self, x):
        emb = self.get_embedding(x)
        logits = self.classifier(emb)
        return logits

class XGBoostWrapper:
    """Wrapper for XGBoost Classifier."""
    def __init__(self):
        self.model = xgb.XGBClassifier(
            n_estimators=300,
            max_depth=6,
            learning_rate=0.05,
            subsample=0.8,
            colsample_bytree=0.8,
            eval_metric='logloss',
            use_label_encoder=False
        )
        
    def fit(self, X, y):
        self.model.fit(X, y)
        
    def predict(self, X):
        return self.model.predict(X)
        
    def predict_proba(self, X):
        return self.model.predict_proba(X)
        
    def get_booster(self):
        """Returns the underlying booster for TreeSHAP access."""
        return self.model.get_booster()

class LightGBMWrapper:
    """Wrapper for LightGBM Classifier."""
    def __init__(self):
        self.model = lgb.LGBMClassifier(
            n_estimators=300,
            max_depth=6,
            learning_rate=0.05,
            subsample=0.8,
            colsample_bytree=0.8,
            metric='binary_logloss',
            verbose=-1
        )
        
    def fit(self, X, y):
        self.model.fit(X, y)
        
    def predict(self, X):
        return self.model.predict(X)
        
    def predict_proba(self, X):
        return self.model.predict_proba(X)
        
    def get_booster(self):
        """Returns the underlying booster for TreeSHAP access."""
        return self.model.booster_
