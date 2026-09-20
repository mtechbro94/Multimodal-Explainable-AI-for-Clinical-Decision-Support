import torch
import torch.nn as nn

class LateFusionModel(nn.Module):
    """
    Late-fusion multimodal baseline.
    Combines tabular and imaging embeddings prior to a joint classification head.
    """
    def __init__(self, tabular_encoder, image_encoder, tab_emb_dim=64, img_emb_dim=1024):
        super().__init__()
        self.tabular_encoder = tabular_encoder
        self.image_encoder = image_encoder
        
        # Disable the standalone classifiers in encoders if necessary
        # Assuming we just call their feature extraction methods
        
        self.fusion_head = nn.Sequential(
            nn.Linear(tab_emb_dim + img_emb_dim, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 1)
        )
        
        self._init_weights()

    def _init_weights(self):
        for m in self.fusion_head.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm1d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def get_modality_embeddings(self, x_tab, x_img):
        """
        Returns individual embeddings for attribution.
        """
        tab_emb = self.tabular_encoder.get_embedding(x_tab)
        # Handle DenseNet vs ViT signature
        if hasattr(self.image_encoder, 'get_features'):
            img_emb = self.image_encoder.get_features(x_img)
        else:
            img_emb = self.image_encoder.get_embedding(x_img)
            
        return tab_emb, img_emb

    def forward(self, x_tab, x_img):
        tab_emb, img_emb = self.get_modality_embeddings(x_tab, x_img)
        fused_emb = torch.cat([tab_emb, img_emb], dim=1)
        logits = self.fusion_head(fused_emb)
        return logits
