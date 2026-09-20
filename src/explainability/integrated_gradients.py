import torch
import numpy as np

class IntegratedGradients:
    def __init__(self, model, baseline=None):
        self.model = model
        self.baseline = baseline
        
    def attribute(self, input_tensor, n_steps=50, target_class=None):
        if self.baseline is None:
            self.baseline = torch.zeros_like(input_tensor)
            
        # Create interpolated inputs
        alphas = torch.linspace(0.0, 1.0, n_steps + 1).to(input_tensor.device)
        alphas = alphas.view(-1, *([1] * (input_tensor.dim() - 1)))
        
        interpolated_inputs = self.baseline + alphas * (input_tensor - self.baseline)
        interpolated_inputs.requires_grad_(True)
        
        # Compute gradients
        gradients = []
        for i in range(n_steps + 1):
            inp = interpolated_inputs[i:i+1]
            output = self.model(inp)
            
            if target_class is None:
                target = output.argmax(dim=1)
            else:
                target = torch.tensor([target_class]).to(input_tensor.device)
                
            score = output[0, target]
            
            self.model.zero_grad()
            score.backward()
            gradients.append(inp.grad.detach())
            
        gradients = torch.cat(gradients, dim=0)
        
        # Riemann sum approximation
        mean_gradients = torch.mean(gradients[:-1], dim=0, keepdim=True)
        attributions = (input_tensor - self.baseline) * mean_gradients
        
        return attributions

    def attribute_tabular(self, x_tab, model_forward_fn, n_steps=50):
        # Wrapper for tabular
        self.model = model_forward_fn
        return self.attribute(x_tab, n_steps)
        
    def attribute_image(self, x_img, model_forward_fn, n_steps=50):
        # Wrapper for image
        self.model = model_forward_fn
        return self.attribute(x_img, n_steps)
