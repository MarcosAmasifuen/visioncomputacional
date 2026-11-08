import torch
import torch.nn as nn
import timm


class AnomalyDetector(nn.Module):
    """
    Cabezal de clasificación multiclase utilizando Transfer Learning sobre un ViT pre-entrenado.
    """

    def __init__(self, num_classes=3, **kwargs):
        super().__init__()

        # 1. Descargamos un ViT "Tiny" (ligero y rápido) pre-entrenado con millones de imágenes
        self.backbone = timm.create_model('vit_tiny_patch16_224', pretrained=True)

        # 2. Extraemos la dimensión de salida del modelo base (usualmente 192 para la versión tiny)
        embed_dim = self.backbone.head.in_features

        # 3. Quitamos el clasificador original de 1000 clases que trae por defecto
        self.backbone.head = nn.Identity()

        # 4. Colocamos nuestro propio cabezal de Serenazgo (3 clases)
        self.classifier = nn.Sequential(
            nn.Linear(embed_dim, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, num_classes)
        )

    def forward(self, x, return_attention=False):
        # El backbone extrae el contexto global de la imagen
        features = self.backbone(x)

        # Nuestro clasificador toma la decisión de Seguridad (Normal, Violencia, Arma)
        anomaly_score = self.classifier(features)

        # Mantenemos la estructura de retorno para compatibilidad con el resto del código
        if return_attention:
            return anomaly_score, None
        return anomaly_score