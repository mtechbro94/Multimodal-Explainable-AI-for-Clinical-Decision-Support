import shap
import numpy as np
import torch

class TreeSHAPExplainer:
    def __init__(self, model):
        self.model = model
        self.explainer = shap.TreeExplainer(self.model)
        
    def explain(self, X):
        return self.explainer.shap_values(X)
        
    def explain_single(self, x):
        return self.explainer.shap_values(x.reshape(1, -1))
        
    def get_feature_importance(self):
        # Assumes explain() has been called or uses internal properties depending on model
        return None

class KernelSHAPExplainer:
    def __init__(self, model_fn, background_data, n_background=100):
        self.model_fn = model_fn
        if len(background_data) > n_background:
            idx = np.random.choice(len(background_data), n_background, replace=False)
            self.background_data = background_data[idx]
        else:
            self.background_data = background_data
            
        self.explainer = shap.KernelExplainer(self.model_fn, self.background_data)
        
    def explain(self, X, nsamples=500):
        return self.explainer.shap_values(X, nsamples=nsamples)
        
    def explain_single(self, x, nsamples=500):
        return self.explainer.shap_values(x.reshape(1, -1), nsamples=nsamples)

class MultimodalSHAPExplainer:
    def __init__(self, multimodal_model, background_tab, background_img):
        self.model = multimodal_model
        self.background_tab = background_tab
        self.background_img = background_img
        
    def explain_tabular_contribution(self, x_tab, x_img):
        def model_fn_tab(x_tab_in):
            x_tab_tensor = torch.tensor(x_tab_in, dtype=torch.float32)
            x_img_tensor = x_img.expand(x_tab_tensor.shape[0], -1, -1, -1)
            with torch.no_grad():
                return self.model(x_tab_tensor, x_img_tensor).numpy()
                
        explainer = KernelSHAPExplainer(model_fn_tab, self.background_tab.numpy())
        return explainer.explain_single(x_tab.numpy())
        
    def explain_image_contribution(self, x_tab, x_img, superpixel_segments=50):
        # Requires advanced segmentation not fully implemented in a small stub,
        # but follows KernelSHAP pattern over superpixels
        return None
