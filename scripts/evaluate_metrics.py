import os
import torch
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from data.transforms import get_val_transforms
from matplotlib.gridspec import GridSpec
import seaborn as sns
from tqdm import tqdm
from sklearn.metrics import (confusion_matrix, classification_report,
                             roc_curve, auc, accuracy_score, precision_recall_fscore_support)
from sklearn.preprocessing import label_binarize
import warnings

warnings.filterwarnings('ignore')

# Importar tu arquitectura ViT-3D
from models.spatiotemporal_vit import SpatioTemporalViT


# NOTA: Importa aquí tu Dataset y DataLoader reales
# from data.dataset import VideoDataset
# from data.transforms import get_val_transforms
# from torch.utils.data import DataLoader

class EvaluadorDoctoralReal:
    def __init__(self, model_path, seq_length=5, device=None):
        self.device = device if device else torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.seq_length = seq_length
        self.classes = ['NORMAL', 'VIOLENCIA', 'ARMA']
        self.n_classes = len(self.classes)

        print(f"[*] Iniciando Evaluación Cuantitativa DSR en dispositivo: {self.device}")
        print(f"[*] Cargando pesos del modelo desde: {model_path}")

        # 1. Inicializar y cargar modelo real
        self.model = SpatioTemporalViT(num_classes=self.n_classes, num_frames=seq_length).to(self.device)
        self.model.load_state_dict(torch.load(model_path, map_location=self.device, weights_only=True))
        self.model.eval()

    def evaluar_dataset(self, test_loader):
        y_true = []
        y_pred = []
        y_prob = []  # Crucial para las curvas ROC

        print("[*] Ejecutando inferencia sobre el conjunto de prueba SCVD (Test Set)...")

        with torch.no_grad():
            for videos, labels in tqdm(test_loader, desc="Evaluando Lotes de Video"):
                # Asegurar tensores en GPU/CPU
                videos = videos.to(self.device)
                labels = labels.to(self.device)

                # Inferencia del ViT-3D
                outputs = self.model(videos)

                # Extraer Probabilidades (Softmax) y Predicción Final (Argmax)
                probs = torch.softmax(outputs, dim=1)
                _, predicted = torch.max(probs, 1)

                # Almacenar para el análisis estadístico
                y_true.extend(labels.cpu().numpy())
                y_pred.extend(predicted.cpu().numpy())
                y_prob.extend(probs.cpu().numpy())

        # Transformar a arrays de NumPy
        y_true = np.array(y_true)
        y_pred = np.array(y_pred)
        y_prob = np.array(y_prob)

        self._generar_dashboard_doctoral(y_true, y_pred, y_prob)

    def _generar_dashboard_doctoral(self, y_true, y_pred, y_prob):
        num_samples = len(y_true)
        acc = accuracy_score(y_true, y_pred)
        precision, recall, f1, support = precision_recall_fscore_support(y_true, y_pred)

        print("\n" + "=" * 60)
        print(f" EVALUACIÓN FINALIZADA - ACCURACY REAL: {acc:.4%} ".center(60, "="))
        print("=" * 60)

        # ========================================================
        # RENDERIZADO DEL DASHBOARD (Motor Constrained Layout)
        # ========================================================
        fig = plt.figure(figsize=(10, 11), dpi=100, constrained_layout=True)
        gs = GridSpec(2, 2, figure=fig, height_ratios=[1.2, 1])

        fig.suptitle(f'Evaluación Cuantitativa DSR: Artefacto ViT-3D Late-Fusion\n(Accuracy Global: {acc:.2%})',
                     fontsize=16, fontweight='bold')

        # --- 1. MATRIZ DE CONFUSIÓN ---
        ax1 = fig.add_subplot(gs[0, 0])
        cm = confusion_matrix(y_true, y_pred)
        cm_norm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

        sns.heatmap(cm_norm, annot=True, fmt=".2%", cmap="Blues",
                    xticklabels=self.classes, yticklabels=self.classes,
                    cbar=False, annot_kws={"size": 11, "weight": "bold"}, ax=ax1)

        ax1.set_title('Matriz de Confusión (Recall)', fontsize=13, fontweight='bold', pad=10)
        ax1.set_ylabel('Etiqueta Real', fontsize=11, fontweight='bold')
        ax1.set_xlabel('Predicción del Modelo', fontsize=11, fontweight='bold')
        ax1.tick_params(axis='y', rotation=0, labelsize=10)
        ax1.tick_params(axis='x', labelsize=10)

        # --- 2. CURVAS ROC MULTICLASE ---
        ax2 = fig.add_subplot(gs[0, 1])
        y_bin = label_binarize(y_true, classes=[0, 1, 2])
        colores = ['#2ecc71', '#f39c12', '#e74c3c']

        for i, color in zip(range(self.n_classes), colores):
            # Proteger contra clases faltantes en datasets pequeños
            if np.sum(y_bin[:, i]) > 0:
                fpr, tpr, _ = roc_curve(y_bin[:, i], y_prob[:, i])
                ax2.plot(fpr, tpr, color=color, lw=2, label=f'ROC {self.classes[i]} (AUC={auc(fpr, tpr):.4f})')

        ax2.plot([0, 1], [0, 1], 'k--', lw=1.5, alpha=0.5)
        ax2.set_xlim([-0.01, 1.0])
        ax2.set_ylim([0.0, 1.05])
        ax2.set_xlabel('FPR (Falsos Positivos)', fontsize=11, fontweight='bold')
        ax2.set_ylabel('TPR (Verdaderos Positivos)', fontsize=11, fontweight='bold')
        ax2.set_title('Curva ROC One-vs-Rest', fontsize=13, fontweight='bold', pad=10)
        ax2.legend(loc="lower right", fontsize=10)
        ax2.grid(True, linestyle=':', alpha=0.6)

        # --- 3. TABLA DE MÉTRICAS ---
        ax3 = fig.add_subplot(gs[1, :])
        ax3.axis('off')

        cell_text = []
        for i in range(self.n_classes):
            cell_text.append([f"{precision[i]:.4f}", f"{recall[i]:.4f}", f"{f1[i]:.4f}", str(support[i])])

        cell_text.append(["---", "---", "---", "---"])
        cell_text.append(
            [f"{np.mean(precision):.4f}", f"{np.mean(recall):.4f}", f"{np.mean(f1):.4f}", str(num_samples)])

        filas = self.classes + ['---------', 'MACRO AVG']
        columnas = ['Precision', 'Recall', 'F1-Score', 'Soporte']

        tabla = ax3.table(cellText=cell_text, rowLabels=filas, colLabels=columnas,
                          loc='center', cellLoc='center', bbox=[0.1, 0.1, 0.8, 0.8])

        tabla.auto_set_font_size(False)
        tabla.set_fontsize(11)
        ax3.set_title('Reporte de Clasificación Cuantitativa (F1-Score)', fontsize=13, fontweight='bold', pad=10)

        for (row, col), cell in tabla.get_celld().items():
            cell.set_edgecolor('#dddddd')
            if row == 0 or col == -1:
                cell.set_text_props(weight='bold')

        for j in range(len(columnas)):
            tabla[(0, j)].set_facecolor('#f4f6f9')
        for i in range(1, len(filas) + 1):
            tabla[(i, -1)].set_facecolor('#f4f6f9')

        plt.savefig("evaluacion_real_vit3d.png", dpi=300)
        print("[*] Artefacto visual generado y guardado como 'evaluacion_real_vit3d.png'.")
        plt.show()


if __name__ == "__main__":
    # Definir rutas base
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

    # -------------------------------------------------------------------
    # RUTA HACIA EL PESO DEL MODELO REAL
    # -------------------------------------------------------------------
    ruta_modelo = os.path.join(base_dir, "models", "spatiotemporal_vit_epoch_8.pth")

    if os.path.exists(ruta_modelo):
        evaluador = EvaluadorDoctoralReal(model_path=ruta_modelo, seq_length=5)

        # -------------------------------------------------------------------
        #  DATALOADER REAL
        # Ejemplo de cómo deberías instanciarlo:
        #
        # from data.dataset import VideoDataset
        # from data.transforms import get_val_transforms
        # from torch.utils.data import DataLoader
        #
        # transformaciones = get_val_transforms(img_size=224)
        # ruta_dataset_test = os.path.join(base_dir, "dataset", "test")
        # dataset_prueba = VideoDataset(root_dir=ruta_dataset_test, seq_length=5, transform=transformaciones)
        # test_dataloader = DataLoader(dataset_prueba, batch_size=8, shuffle=False, num_workers=2)
        # -------------------------------------------------------------------

        # Borra este 'None' y pon el nombre de tu variable dataloader (ej. test_dataloader)
        mi_dataloader_real = None

        if mi_dataloader_real is not None:
            evaluador.evaluar_dataset(mi_dataloader_real)
        else:
            print("\n[!] ALERTA DE SISTEMA: El evaluador está listo.")
            print("[!] Por favor, comenta la línea 'mi_dataloader_real = None' y")
            print("[!] conecta tu DataLoader real de PyTorch en la línea 170.")
    else:
        print(f"[!] ERROR CRÍTICO: No se encontró el archivo de pesos del modelo en:\n{ruta_modelo}")