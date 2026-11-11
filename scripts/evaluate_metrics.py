import os
import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score

# Importar tu arquitectura
from models.spatiotemporal_vit import SpatioTemporalViT


# NOTA: Importa aquí tu DataLoader del dataset de prueba (SCVD Test Set)
# from data.dataset import get_test_dataloader

class EvaluadorDoctoral:
    def __init__(self, model_path, seq_length=5, device=None):
        self.device = device if device else torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.seq_length = seq_length
        self.classes = ['NORMAL', 'VIOLENCIA', 'ARMA']

        print(f"[*] Iniciando Evaluación Cuantitativa en dispositivo: {self.device}")

        # Inicializar y cargar modelo
        self.model = SpatioTemporalViT(num_classes=3, num_frames=seq_length).to(self.device)
        self.model.load_state_dict(torch.load(model_path, map_location=self.device, weights_only=True))
        self.model.eval()

    def evaluar_dataset(self, test_loader):
        y_true = []
        y_pred = []

        print("[*] Ejecutando inferencia sobre el conjunto de prueba (Test Set)...")

        with torch.no_grad():
            for videos, labels in tqdm(test_loader, desc="Evaluando Lotes"):
                # videos shape expected: [Batch, Frames, Channels, Height, Width]
                videos = videos.to(self.device)
                labels = labels.to(self.device)

                outputs = self.model(videos)
                _, predicted = torch.max(outputs, 1)

                y_true.extend(labels.cpu().numpy())
                y_pred.extend(predicted.cpu().numpy())

        self._generar_reporte_y_matriz(np.array(y_true), np.array(y_pred))

    def _generar_reporte_y_matriz(self, y_true, y_pred):
        # 1. Calcular Métricas Cuantitativas
        acc = accuracy_score(y_true, y_pred)
        reporte = classification_report(y_true, y_pred, target_names=self.classes, digits=4)

        print("\n" + "=" * 50)
        print(" REPORTE DE MÉTRICAS (ESTÁNDAR IEEE) ".center(50, "="))
        print("=" * 50)
        print(f"Accuracy Global (Precisión): {acc:.4%}\n")
        print(reporte)
        print("=" * 50)

        # 2. Generar Matriz de Confusión
        cm = confusion_matrix(y_true, y_pred)

        # Normalizar matriz para mostrar porcentajes por fila (Recall)
        cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

        # 3. Renderizado de Gráfica Académica (Alta Resolución)
        plt.figure(figsize=(8, 6), dpi=300)  # DPI 300 para impresión en tesis
        sns.set_theme(style="whitegrid", font_scale=1.1)

        heatmap = sns.heatmap(cm_normalized, annot=True, fmt=".2%", cmap="Blues",
                              xticklabels=self.classes, yticklabels=self.classes,
                              cbar=True, annot_kws={"size": 12, "weight": "bold"})

        # Estilos de ejes
        heatmap.set_xticklabels(heatmap.get_xticklabels(), rotation=0, fontweight='bold')
        heatmap.set_yticklabels(heatmap.get_yticklabels(), rotation=45, fontweight='bold')

        plt.title('Matriz de Confusión Normalizada\n(ViT-3D Late-Fusion sobre Dataset SCVD)',
                  fontsize=14, fontweight='bold', pad=15)
        plt.ylabel('Etiqueta Verdadera (Ground Truth)', fontsize=12, fontweight='bold')
        plt.xlabel('Predicción del Modelo Espacio-Temporal', fontsize=12, fontweight='bold')
        plt.tight_layout()

        # Guardar artefacto visual
        output_filename = "matriz_confusion_vit3d.png"
        plt.savefig(output_filename, bbox_inches='tight')
        print(f"\n[*] Artefacto visual guardado exitosamente como: '{output_filename}'")
        plt.show()


if __name__ == "__main__":
    # Rutas
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    ruta_modelo = os.path.join(base_dir, "models", "spatiotemporal_vit_epoch_8.pth")

    if os.path.exists(ruta_modelo):
        evaluador = EvaluadorDoctoral(model_path=ruta_modelo, seq_length=5)

        # AQUÍ: Reemplazar 'None' con la instancia real de tu DataLoader de prueba
        test_dataloader = None

        if test_dataloader is not None:
            evaluador.evaluar_dataset(test_dataloader)
        else:
            print("[!] ERROR: Debes instanciar el 'test_dataloader' en la línea 77 para ejecutar la prueba.")
    else:
        print(f"[!] ERROR: No se encontró el modelo en la ruta {ruta_modelo}")