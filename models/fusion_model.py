# ============================================================
# models/fusion_model.py  --  Modelo que fusiona CNN + características botánicas
# ============================================================
import torch
import torch.nn as nn
from torchvision.models import resnet18, ResNet18_Weights

from config.settings import DROPOUT_RATE

class FusionCoffeeModel(nn.Module):
    """
    Modelo híbrido:
      - ResNet18 para extraer características visuales de la imagen.
      - Vector de características botánicas (extraídas manualmente).
      - Fusión mediante concatenación y capa fully connected final.
    """
    def __init__(self, num_classes: int, bot_feat_dim: int, pretrained: bool = True):
        super().__init__()
        # Backbone CNN
        if pretrained:
            weights = ResNet18_Weights.IMAGENET1K_V1
            self.backbone = resnet18(weights=weights)
        else:
            self.backbone = resnet18(weights=None)
        # Quitamos la última capa fc de ResNet para obtener el embedding
        in_features = self.backbone.fc.in_features
        self.backbone.fc = nn.Identity()   # ahora devuelve el vector de 512

        # Congelar TODAS las capas inicialmente
        for param in self.backbone.parameters():
            param.requires_grad = False

        # Descongelar layer3 y layer4 para fine‑tuning más profundo
        # for param in self.backbone.layer3.parameters():
        #     param.requires_grad = True
        for param in self.backbone.layer4.parameters():
            param.requires_grad = True

        # Capa de fusión: concatena embedding (512) + bot_feat_dim
        fusion_dim = in_features + bot_feat_dim
        self.classifier = nn.Sequential(
            nn.Linear(fusion_dim, 256),
            nn.ReLU(),
            nn.Dropout(DROPOUT_RATE),
            nn.Linear(256, num_classes)
        )

    def forward(self, x: torch.Tensor, bot_feats: torch.Tensor) -> torch.Tensor:
        # x: [batch, 3, H, W] (imagen normalizada)
        # bot_feats: [batch, bot_feat_dim]
        visual_embed = self.backbone(x)          # [batch, 512]
        combined = torch.cat([visual_embed, bot_feats], dim=1)  # [batch, 512+bot_feat_dim]
        return self.classifier(combined)