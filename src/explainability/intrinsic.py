import torch
import torch.autograd as autograd

class IntrinsicConceptExplainer:
    def __init__(self, model):
        self.model = model
        
    def explain(self, x_tab, x_img):
        self.model.eval()
        
        # Forward pass returning concepts and predictors
        # Assuming the model returns: output, concepts (dict of tab, img, shared), weights
        output, concepts, weights = self.model(x_tab, x_img, return_concepts_and_weights=True)
        
        tab_concepts = concepts['tab']
        img_concepts = concepts['img']
        shared_concepts = concepts['shared']
        
        tab_weights = weights['tab']
        img_weights = weights['img']
        shared_weights = weights['shared']
        
        # Attribution = weight * activation
        tab_concept_attributions = tab_concepts * tab_weights
        img_concept_attributions = img_concepts * img_weights
        shared_concept_attributions = shared_concepts * shared_weights
        
        return {
            'tab_concept_attributions': tab_concept_attributions.detach().cpu(),
            'img_concept_attributions': img_concept_attributions.detach().cpu(),
            'shared_concept_attributions': shared_concept_attributions.detach().cpu(),
            'cross_attention_weights': None # Extract if available
        }
        
    def explain_tabular_features(self, x_tab, x_img):
        x_tab.requires_grad_(True)
        output, concepts, weights = self.model(x_tab, x_img, return_concepts_and_weights=True)
        tab_concepts = concepts['tab']
        tab_weights = weights['tab']
        
        total_tab_attribution = torch.sum(tab_concepts * tab_weights)
        
        # Autograd chain rule
        total_tab_attribution.backward()
        
        return x_tab.grad.detach().cpu()
        
    def explain_image_regions(self, x_tab, x_img, spatial_size=7):
        x_img.requires_grad_(True)
        output, concepts, weights = self.model(x_tab, x_img, return_concepts_and_weights=True)
        
        img_concepts = concepts['img']
        img_weights = weights['img']
        
        total_img_attribution = torch.sum(img_concepts * img_weights)
        total_img_attribution.backward()
        
        # Simplistic return of image gradients, actual spatial attribution requires hook on spatial maps
        grad = x_img.grad.detach()
        spatial_map = torch.mean(torch.abs(grad), dim=1) # Average over channels
        spatial_map = torch.nn.functional.interpolate(spatial_map.unsqueeze(1), size=(224, 224), mode='bilinear')
        
        return spatial_map.squeeze(1).cpu()
        
    def completeness_check(self, x_tab, x_img):
        self.model.eval()
        with torch.no_grad():
            output, concepts, weights = self.model(x_tab, x_img, return_concepts_and_weights=True)
            
            total_concept_sum = (
                torch.sum(concepts['tab'] * weights['tab'], dim=1) +
                torch.sum(concepts['img'] * weights['img'], dim=1) +
                torch.sum(concepts['shared'] * weights['shared'], dim=1)
            )
            
            # Assuming output is logit, adding bias
            bias = self.model.predictor.bias if hasattr(self.model, 'predictor') and hasattr(self.model.predictor, 'bias') else 0
            predicted_logit = total_concept_sum + bias
            
            ratio = total_concept_sum / (output - bias + 1e-8)
            
        return ratio.cpu().numpy()
