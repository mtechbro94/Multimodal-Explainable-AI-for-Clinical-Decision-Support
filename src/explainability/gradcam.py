import torch
import torch.nn.functional as F
import numpy as np

class GradCAM:
    def __init__(self, model, target_layer):
        self.model = model
        self.target_layer = target_layer
        self.gradients = None
        self.activations = None
        
        self.target_layer.register_forward_hook(self.save_activation)
        self.target_layer.register_full_backward_hook(self.save_gradient)
        
    def save_activation(self, module, input, output):
        self.activations = output
        
    def save_gradient(self, module, grad_input, grad_output):
        self.gradients = grad_output[0]
        
    def generate(self, input_tensor, target_class=None):
        self.model.eval()
        self.model.zero_grad()
        
        output = self.model(input_tensor)
        if target_class is None:
            target_class = output.argmax(dim=1).item()
            
        target = output[0][target_class]
        target.backward()
        
        gradients = self.gradients.detach()
        activations = self.activations.detach()
        
        # Global average pool gradients
        pooled_gradients = torch.mean(gradients, dim=[0, 2, 3])
        
        # Weight activations
        for i in range(activations.shape[1]):
            activations[:, i, :, :] *= pooled_gradients[i]
            
        heatmap = torch.mean(activations, dim=1).squeeze()
        
        # ReLU
        heatmap = F.relu(heatmap)
        
        # Normalize to [0,1]
        heatmap /= torch.max(heatmap) + 1e-8
        
        return heatmap.cpu().numpy()

    @staticmethod
    def overlay_on_image(image_tensor, heatmap, alpha=0.4):
        import matplotlib.pyplot as plt
        from matplotlib import cm
        
        if torch.is_tensor(image_tensor):
            img = image_tensor.cpu().numpy().transpose(1, 2, 0)
        else:
            img = image_tensor
            
        # Normalize image to [0,1]
        img = (img - img.min()) / (img.max() - img.min())
        
        # Resize heatmap
        import cv2
        heatmap = cv2.resize(heatmap, (img.shape[1], img.shape[0]))
        
        # Apply colormap
        cmap = cm.get_cmap('jet')
        heatmap_colored = cmap(heatmap)[:, :, :3]
        
        overlay = alpha * heatmap_colored + (1 - alpha) * img
        overlay = np.clip(overlay, 0, 1)
        
        return overlay
