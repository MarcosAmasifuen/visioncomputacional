import torch
import torch.nn as nn
from models.vit_backbone import ViTBackbone


class AnomalyDetector(nn.Module):
    """
    Cabezal de clasificación que toma el ViT y decide si el frame CCTV contiene un patrón de riesgo.
    """

    def __init__(self, embed_dim=768, num_classes=1, **kwargs):
        super().__init__()
        # 1. Inicializamos el Backbone (el extractor de características visuales)
        self.backbone = ViTBackbone(embed_dim=embed_dim, **kwargs)

        # 2. Cabezal de clasificación: toma el resumen de la imagen (token [CLS])
        # y lo comprime hasta obtener un solo valor numérico.
        self.classifier = nn.Sequential(
            nn.Linear(embed_dim, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, num_classes),
            nn.Sigmoid()  # Función Sigmoide para forzar la salida a un porcentaje entre 0.0 y 1.0
        )

    def forward(self, x, return_attention=False):
        """
        Entrada: Tensor de video -> (Batch, Canales, Alto, Ancho)
        Salida: Probabilidad de anomalía [0, 1] y mapas de atención para explicabilidad.
        """
        # A. Procesamos los frames a través del Transformer
        if return_attention:
            features, attn_maps = self.backbone(x, return_attention=True)
        else:
            features = self.backbone(x)
            attn_maps = None

        # B. Extraemos SOLO el token [CLS], que concentra el contexto global.
        # Está ubicado en la dimensión 1, índice 0 -> features[:, 0, :]
        cls_token = features[:, 0, :]

        # C. Predicción de la anomalía
        anomaly_score = self.classifier(cls_token)

        if return_attention:
            return anomaly_score, attn_maps
        return anomaly_score


# ==========================================
# Prueba del Detector de Anomalías Completo
# ==========================================
if __name__ == "__main__":
    # Simulamos el monitoreo de 4 cámaras simultáneas
    dummy_cctv_frames = torch.randn(4, 3, 224, 224)

    # Instanciamos el modelo completo (ViT con profundidad 4 para que sea ligero en la prueba local)
    detector = AnomalyDetector(depth=4)

    # Ejecutamos la inferencia simulada
    scores, attention_maps = detector(dummy_cctv_frames, return_attention=True)

    print(f"Cámaras analizadas (Batch): {dummy_cctv_frames.shape[0]}")
    print(f"\nProbabilidad de delito (Umbral típico > 0.75):")
    for i, score in enumerate(scores):
        print(f" - Cámara {i + 1}: {score.item():.2%}")

    print(f"\n¿Mapas de atención listos para UI?: {'Sí' if attention_maps else 'No'}")
    print(f"Formato de salida del score: {scores.shape} -> (Batch, Num_Clases)")