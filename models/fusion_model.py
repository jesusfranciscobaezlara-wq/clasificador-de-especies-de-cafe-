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

        # Solo se descongela layer4: permite adaptar las features visuales
        # de más alto nivel a hojas de café sin reentrenar toda la red
        # preentrenada en ImageNet (más rápido y con menos riesgo de overfitting).
        for param in self.backbone.layer4.parameters():
            param.requires_grad = True

        # Las botánicas llegan en escalas muy distintas entre sí (ratios ~0-1,
        # conteos de píxeles, grados) y frente al embedding visual ya normalizado
        # por ImageNet; sin esto podían dominar o quedar opacadas en la fusión.
        self.bot_norm = nn.BatchNorm1d(bot_feat_dim)

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
        bot_normed = self.bot_norm(bot_feats)
        combined = torch.cat([visual_embed, bot_normed], dim=1)  # [batch, 512+bot_feat_dim]
        return self.classifier(combined)