import os
import glob
import cv2
import torch
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import seaborn as sns
from tqdm import tqdm
from PIL import Image
from sklearn.metrics import (confusion_matrix, classification_report,
                             roc_curve, auc, accuracy_score, precision_recall_fscore_support)
from sklearn.preprocessing import label_binarize
import warnings

warnings.filterwarnings('ignore')

from torch.utils.data import Dataset, DataLoader
from models.spatiotemporal_vit import SpatioTemporalViT
from data.transforms import get_val_transforms


# =====================================================================
# 1. MOTOR DE LECTURA DE VIDEOS (Basado en la captura de tu disco duro)
# =====================================================================
class LectorVideosCCTV(Dataset):
    def __init__(self, root_dir, seq_length=5, transform=None):
        self.root_dir = root_dir
        self.seq_length = seq_length
        self.transform = transform

        # Mapeo exacto de las carpetas de tu captura de pantalla
        self.class_map = {
            'Normal': 0,
            'Violence': 1,
            'Weaponized': 2
        }

        self.video_paths = []
        self.labels = []

        print(f"[*] Escaneando disco duro en: {self.root_dir}")
        for class_name, class_idx in self.class_map.items():
            class_dir = os.path.join(self.root_dir, class_name)
            if os.path.exists(class_dir):
                # Buscar mp4, avi, mkv
                videos = glob.glob(os.path.join(class_dir, "*.*"))
                valid_videos = [v for v in videos if v.endswith(('.mp4', '.avi', '.mkv'))]
                for v in valid_videos:
                    self.video_paths.append(v)
                    self.labels.append(class_idx)
                print(f"    - [{class_name}]: {len(valid_videos)} videos encontrados.")
            else:
                print(f"    [!] ADVERTENCIA: No se encontró la carpeta {class_name}")

        print(f"[*] Total de videos listos para evaluación: {len(self.video_paths)}")

    def __len__(self):
        return len(self.video_paths)

    def __getitem__(self, idx):
        video_path = self.video_paths[idx]
        label = self.labels[idx]

        cap = cv2.VideoCapture(video_path)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        frames_capturados = []

        if total_frames > 0:
            # Extraer 5 fotogramas distribuidos uniformemente
            indices = np.linspace(0, total_frames - 1, self.seq_length, dtype=int)
            for i in indices:
                cap.set(cv2.CAP_PROP_POS_FRAMES, i)
                ret, frame = cap.read()
                if ret:
                    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    frames_capturados.append(frame)
                else:
                    frames_capturados.append(np.zeros((224, 224, 3), dtype=np.uint8))
        cap.release()

        # Rellenar con negro si el video está corrupto o es muy corto
        while len(frames_capturados) < self.seq_length:
            frames_capturados.append(np.zeros((224, 224, 3), dtype=np.uint8))

        # Aplicar transformaciones de PyTorch a cada fotograma
        tensores = []
        for frame in frames_capturados:
            if self.transform:
                # Enviamos el 'frame' crudo (NumPy) porque get_val_transforms
                # ya se encarga de convertirlo a PIL internamente
                tensor = self.transform(frame)
                tensores.append(tensor)

        # Apilar temporalmente: shape final [seq_length, channels, H, W]
        seq_tensor = torch.stack(tensores)
        return seq_tensor, label


# =====================================================================
# 2. MOTOR DE EVALUACIÓN CUANTITATIVA DSR
# =====================================================================
class EvaluadorDoctoralReal:
    def __init__(self, model_path, seq_length=5, device=None):
        self.device = device if device else torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.seq_length = seq_length
        self.classes = ['NORMAL', 'VIOLENCIA', 'ARMA']
        self.n_classes = len(self.classes)

        print(f"\n[*] INICIANDO INFERENCIA EN DISPOSITIVO: {self.device}")

        self.model = SpatioTemporalViT(num_classes=self.n_classes, num_frames=seq_length).to(self.device)
        self.model.load_state_dict(torch.load(model_path, map_location=self.device, weights_only=True))
        self.model.eval()

    def evaluar_dataset(self, test_loader):
        y_true = []
        y_pred = []
        y_prob = []

        with torch.no_grad():
            for videos, labels in tqdm(test_loader, desc="Clasificando Videos", unit="batch"):
                videos = videos.to(self.device)
                labels = labels.to(self.device)

                outputs = self.model(videos)
                probs = torch.softmax(outputs, dim=1)
                _, predicted = torch.max(probs, 1)

                y_true.extend(labels.cpu().numpy())
                y_pred.extend(predicted.cpu().numpy())
                y_prob.extend(probs.cpu().numpy())

        y_true = np.array(y_true)
        y_pred = np.array(y_pred)
        y_prob = np.array(y_prob)

        self._generar_dashboard_doctoral(y_true, y_pred, y_prob)

    def _generar_dashboard_doctoral(self, y_true, y_pred, y_prob):
        num_samples = len(y_true)
        acc = accuracy_score(y_true, y_pred)
        precision, recall, f1, support = precision_recall_fscore_support(y_true, y_pred, zero_division=0)

        print("\n" + "=" * 60)
        print(f" EVALUACIÓN FINALIZADA - ACCURACY REAL: {acc:.4%} ".center(60, "="))
        print("=" * 60)

        fig = plt.figure(figsize=(10, 11), dpi=100, constrained_layout=True)
        gs = GridSpec(2, 2, figure=fig, height_ratios=[1.2, 1])

        fig.suptitle(f'Evaluación Cuantitativa DSR: Artefacto ViT-3D Late-Fusion\n(Accuracy Global: {acc:.2%})',
                     fontsize=16, fontweight='bold')

        # --- PANEL 1. MATRIZ DE CONFUSIÓN ---
        ax1 = fig.add_subplot(gs[0, 0])
        cm = confusion_matrix(y_true, y_pred, labels=[0, 1, 2])
        cm_norm = np.zeros_like(cm, dtype=float)
        # Normalización segura por si una clase no tiene videos
        for i in range(cm.shape[0]):
            if cm.sum(axis=1)[i] > 0:
                cm_norm[i] = cm[i].astype('float') / cm.sum(axis=1)[i]

        sns.heatmap(cm_norm, annot=True, fmt=".2%", cmap="Blues",
                    xticklabels=self.classes, yticklabels=self.classes,
                    cbar=False, annot_kws={"size": 11, "weight": "bold"}, ax=ax1)

        ax1.set_title('Matriz de Confusión (Recall)', fontsize=13, fontweight='bold', pad=10)
        ax1.set_ylabel('Etiqueta Real (Ground Truth)', fontsize=11, fontweight='bold')
        ax1.set_xlabel('Predicción del Modelo', fontsize=11, fontweight='bold')
        ax1.tick_params(axis='y', rotation=0, labelsize=10)
        ax1.tick_params(axis='x', labelsize=10)

        # --- PANEL 2. CURVAS ROC ---
        ax2 = fig.add_subplot(gs[0, 1])
        y_bin = label_binarize(y_true, classes=[0, 1, 2])
        colores = ['#2ecc71', '#f39c12', '#e74c3c']

        for i, color in zip(range(self.n_classes), colores):
            if np.sum(y_bin[:, i]) > 0:  # Evitar error si no hay videos de armas, por ejemplo
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

        # --- PANEL 3. TABLA DE MÉTRICAS ---
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
        print("[*] Artefacto guardado exitosamente como 'evaluacion_real_vit3d.png'.")
        plt.show()


# =====================================================================
# 3. LANZAMIENTO DEL SCRIPT
# =====================================================================
if __name__ == "__main__":
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    ruta_modelo = os.path.join(base_dir, "models", "spatiotemporal_vit_epoch_8.pth")

    # LA RUTA MÁGICA DE TU CAPTURA DE PANTALLA:
    ruta_dataset_raw = os.path.join(base_dir, "data", "raw")

    if not os.path.exists(ruta_modelo):
        print(f"[!] ERROR: No se encontró el modelo en: {ruta_modelo}")
    elif not os.path.exists(ruta_dataset_raw):
        print(f"[!] ERROR: No se encontró la carpeta de videos en: {ruta_dataset_raw}")
    else:
        # Instanciar el evaluador
        evaluador = EvaluadorDoctoralReal(model_path=ruta_modelo, seq_length=5)

        # Preparar transformaciones de 224x224
        transformaciones = get_val_transforms(img_size=224)

        # Levantar el Lector que creamos arriba apuntando a 'data/raw'
        dataset_real = LectorVideosCCTV(root_dir=ruta_dataset_raw, seq_length=5, transform=transformaciones)

        if len(dataset_real) > 0:
            # Empaquetador DataLoader (batch_size=8 procesa 8 videos a la vez. Bájalo a 4 si falta VRAM).
            dataloader_real = DataLoader(dataset_real, batch_size=8, shuffle=False, num_workers=0)

            # ¡Ejecutar!
            evaluador.evaluar_dataset(dataloader_real)
        else:
            print("[!] ERROR: No se encontraron videos mp4/avi dentro de data/raw")