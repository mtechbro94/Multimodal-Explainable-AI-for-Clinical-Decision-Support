import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models

class CrossModalConceptBottleneck(nn.Module):
    """
    Faithfulness-Constrained Cross-Modal Attentive Concept Bottleneck Model (XM-CBM).
    """
    def __init__(self, tab_input_dim, num_tab_concepts=12, num_img_concepts=12, 
                 num_shared_concepts=8, hidden_dim=256, num_heads=4, dropout=0.3):
        super().__init__()
        
        self.num_tab_concepts = num_tab_concepts
        self.num_img_concepts = num_img_concepts
        self.num_shared_concepts = num_shared_concepts
        
        # 1. Tabular Concept Extractor
        self.tab_extractor = nn.Sequential(
            nn.Linear(tab_input_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, num_tab_concepts),
            nn.Sigmoid()
        )
        
        # 2. Image Concept Extractor (DenseNet121 backbone)
        self.img_backbone = models.densenet121(weights=None).features
        self.img_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.img_extractor = nn.Sequential(
            nn.Linear(1024, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, num_img_concepts),
            nn.Sigmoid()
        )
        
        # 3. Cross-Modal Attention Module
        # Note: assuming num_tab_concepts == num_img_concepts for cross-attention projection
        assert num_tab_concepts == num_img_concepts, "Attention requires equal concept dimensions"
        self.attn = nn.MultiheadAttention(embed_dim=num_tab_concepts, num_heads=num_heads, dropout=dropout)
        
        # Learnable alpha for weighting attended vs original features
        self.alpha = nn.Parameter(torch.tensor(0.5))
        
        # 4. Concept Bottleneck Layer (Shared concepts generation)
        total_concepts = num_tab_concepts + num_img_concepts + num_tab_concepts
        self.shared_cbm = nn.Sequential(
            nn.Linear(total_concepts, num_shared_concepts),
            nn.Sigmoid()
        )
        
        # 5. Predictor
        self.predictor = nn.Linear(num_shared_concepts, 1)
        
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm1d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, x_tab, x_img):
        # Extract individual concepts
        c_tab = self.tab_extractor(x_tab)
        
        img_feats = self.img_backbone(x_img)
        img_feats = self.img_pool(img_feats)
        img_feats = torch.flatten(img_feats, 1)
        c_img = self.img_extractor(img_feats)
        
        # Cross-modal attention (seq_len=1, batch, embed_dim)
        c_tab_seq = c_tab.unsqueeze(0)
        c_img_seq = c_img.unsqueeze(0)
        
        c_tab_attended, attn_weights = self.attn(
            query=c_tab_seq, 
            key=c_img_seq, 
            value=c_img_seq
        )
        c_tab_attended = c_tab_attended.squeeze(0)
        
        # We can also do the reverse if symmetric fusion is desired, but 
        # based on spec: alpha * c_tab_attended + (1-alpha) * c_img_attended 
        # For simplicity, we just use c_tab_attended to represent fused concepts here.
        # Alternatively, run reverse attention:
        c_img_attended, _ = self.attn(
            query=c_img_seq,
            key=c_tab_seq,
            value=c_tab_seq
        )
        c_img_attended = c_img_attended.squeeze(0)
        
        c_fused = self.alpha * c_tab_attended + (1 - self.alpha) * c_img_attended
        
        # Create full concept array
        c_all = torch.cat([c_tab, c_img, c_fused], dim=1)
        
        # Extract shared concepts
        c_shared = self.shared_cbm(c_all)
        
        # Predict
        logits = self.predictor(c_shared)
        
        concept_activations_dict = {
            'tab_concepts': c_tab,
            'img_concepts': c_img,
            'shared_concepts': c_shared,
            'cross_attention_weights': attn_weights,
            'predictor_weights': self.predictor.weight.data
        }
        
        return logits, concept_activations_dict

    def get_concept_attributions(self, x_tab, x_img):
        """
        Returns the contribution of each concept to the final prediction.
        """
        logits, concept_dict = self(x_tab, x_img)
        c_shared = concept_dict['shared_concepts']
        weights = self.predictor.weight.squeeze(0)
        contributions = c_shared * weights
        return contributions

    def get_faithfulness_loss(self, x_tab, x_img, logits, concept_acts):
        """
        Computes faithfulness regularization:
        Forces the model's behavior to match its explanation by observing the 
        change in prediction when concepts are masked.
        """
        c_shared = concept_acts['shared_concepts']
        weights = self.predictor.weight.squeeze(0)
        bias = self.predictor.bias
        
        faithfulness_loss = 0.0
        batch_size = c_shared.size(0)
        
        for i in range(self.num_shared_concepts):
            # Calculate actual prediction change if concept i is zeroed out
            c_shared_masked = c_shared.clone()
            c_shared_masked[:, i] = 0.0
            logits_masked = self.predictor(c_shared_masked)
            
            # Prediction difference
            delta_pred = logits - logits_masked
            
            # Expected contribution based on weights
            expected_contrib = c_shared[:, i] * weights[i]
            
            # MSE between change in prediction and expected contribution
            faithfulness_loss += F.mse_loss(delta_pred.squeeze(), expected_contrib)
            
        return faithfulness_loss / self.num_shared_concepts

    def get_concept_alignment_loss(self, tab_concepts, img_concepts):
        """
        Cross-modal alignment loss to encourage consistent representations.
        """
        cos_sim = F.cosine_similarity(tab_concepts, img_concepts, dim=1)
        alignment_loss = 1.0 - cos_sim.mean()
        return alignment_loss

class XMCBMLoss(nn.Module):
    """
    Custom Loss for the XM-CBM model incorporating prediction, 
    concept sparsity, faithfulness, and alignment.
    """
    def __init__(self, lambda_concept=0.1, lambda_faith=0.5, lambda_align=0.2):
        super().__init__()
        self.lambda_concept = lambda_concept
        self.lambda_faith = lambda_faith
        self.lambda_align = lambda_align
        self.bce = nn.BCEWithLogitsLoss()

    def forward(self, logits, targets, concept_dict, model, x_tab, x_img):
        # Base Prediction Loss
        l_pred = self.bce(logits.view(-1), targets.view(-1))
        
        # Concept Sparsity Penalty (L1 on shared concepts)
        c_shared = concept_dict['shared_concepts']
        l_concept = torch.mean(torch.abs(c_shared))
        
        # Faithfulness Loss
        l_faith = model.get_faithfulness_loss(x_tab, x_img, logits, concept_dict)
        
        # Alignment Loss
        l_align = model.get_concept_alignment_loss(
            concept_dict['tab_concepts'], 
            concept_dict['img_concepts']
        )
        
        total_loss = (l_pred + 
                      self.lambda_concept * l_concept + 
                      self.lambda_faith * l_faith + 
                      self.lambda_align * l_align)
                      
        loss_components_dict = {
            'L_total': total_loss.item(),
            'L_pred': l_pred.item(),
            'L_concept': l_concept.item(),
            'L_faith': l_faith.item(),
            'L_align': l_align.item()
        }
        
        return total_loss, loss_components_dict
