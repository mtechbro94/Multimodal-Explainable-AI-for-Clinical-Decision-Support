import torch
import torch.nn as nn
import torchvision.models as models
import timm

class DenseNet121Classifier(nn.Module):
    """
    Imaging Baseline Model: DenseNet121.
    """
    def __init__(self, num_concepts=128, pretrained=False):
        super().__init__()
        weights = models.DenseNet121_Weights.DEFAULT if pretrained else None
        self.backbone = models.densenet121(weights=weights)
        
        # Replace classification head
        in_features = self.backbone.classifier.in_features # 1024
        self.backbone.classifier = nn.Identity()
        
        self.head = nn.Sequential(
            nn.Linear(in_features, num_concepts),
            nn.ReLU(),
            nn.Linear(num_concepts, 1)
        )

    def get_features(self, x):
        """Returns the features before the final classifier."""
        features = self.backbone.features(x)
        out = nn.functional.relu(features, inplace=True)
        out = nn.functional.adaptive_avg_pool2d(out, (1, 1))
        out = torch.flatten(out, 1)
        return out
        
    def get_cam_target_layer(self):
        """Returns the last conv layer for Grad-CAM."""
        return self.backbone.features[-1]

    def forward(self, x):
        features = self.get_features(x)
        logits = self.head(features)
        return logits

class ViTClassifier(nn.Module):
    """
    Imaging Baseline Model: Vision Transformer.
    """
    def __init__(self, num_concepts=128, pretrained=False):
        super().__init__()
        self.backbone = timm.create_model(
            'vit_base_patch16_224', 
            pretrained=pretrained, 
            num_classes=0
        )
        
        # Add classification head
        self.head = nn.Sequential(
            nn.Linear(768, num_concepts),
            nn.ReLU(),
            nn.Linear(num_concepts, 1)
        )

    def get_embedding(self, x):
        """Returns the embedding before the classification head."""
        return self.backbone(x)
        
    def get_attention_weights(self, x):
        """Returns the attention weights from the ViT blocks."""
        # A hacky way to extract attention maps if using timm
        # Typically requires registering forward hooks
        # Placeholder mechanism assuming hooks are set externally
        pass

    def forward(self, x):
        emb = self.get_embedding(x)
        logits = self.head(emb)
        return logits
