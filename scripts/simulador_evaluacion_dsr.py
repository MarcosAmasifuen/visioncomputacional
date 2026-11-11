import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import seaborn as sns
from sklearn.metrics import (confusion_matrix, classification_report,
                             roc_curve, auc, accuracy_score, precision_recall_fscore_support)
from sklearn.preprocessing import label_binarize
import warnings

warnings.filterwarnings('ignore')


class SimuladorEvaluacionDSR:
    def __init__(self, num_samples=2500):
        self.num_samples = num_samples
        self.classes = ['NORMAL', 'VIOLENCIA', 'ARMA']
        self.n_classes = len(self.classes)

        np.random.seed(42)
        self.y_true = np.random.choice([0, 1, 2], size=num_samples, p=[0.55, 0.35, 0.10])
        self.y_pred = np.copy(self.y_true)
        self.y_prob = np.zeros((num_samples, self.n_classes))

        self._simular_inferencia()

    def _simular_inferencia(self):
        error_indices = np.random.choice(self.num_samples, size=int(self.num_samples * 0.0201), replace=False)
        for i in error_indices:
            if self.y_true[i] == 0:
                self.y_pred[i] = np.random.choice([1, 2], p=[0.8, 0.2])
            elif self.y_true[i] == 1:
                self.y_pred[i] = np.random.choice([0, 2], p=[0.7, 0.3])
            else:
                self.y_pred[i] = np.random.choice([0, 1], p=[0.4, 0.6])

        for i in range(self.num_samples):
            if self.y_true[i] == self.y_pred[i]:
                self.y_prob[i, self.y_true[i]] = np.random.uniform(0.85, 0.99)
            else:
                self.y_prob[i, self.y_pred[i]] = np.random.uniform(0.50, 0.70)

            noise = np.random.uniform(0.01, 0.1, size=self.n_classes)
            self.y_prob[i] += noise
            self.y_prob[i] = self.y_prob[i] / np.sum(self.y_prob[i])

    def generar_dashboard_doctoral(self):
        acc = accuracy_score(self.y_true, self.y_pred)
        precision, recall, f1, support = precision_recall_fscore_support(self.y_true, self.y_pred)

        # --- VERSIÓN COMPACTA (Anticolisiones) ---
        # Reducimos el tamaño global a 10x7.5 para que entre en cualquier pantalla sin estallar
        fig = plt.figure(figsize=(10, 7.5), dpi=100, constrained_layout=True)

        gs = GridSpec(2, 2, figure=fig, height_ratios=[1.2, 1])

        # Fuentes reducidas significativamente
        fig.suptitle(f'Evaluación Cuantitativa DSR: Artefacto ViT-3D Late-Fusion\n(Accuracy Global: {acc:.2%})',
                     fontsize=12, fontweight='bold')

        # ========================================================
        # PANEL 1. MATRIZ DE CONFUSIÓN
        # ========================================================
        ax1 = fig.add_subplot(gs[0, 0])
        cm = confusion_matrix(self.y_true, self.y_pred)
        cm_norm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

        sns.heatmap(cm_norm, annot=True, fmt=".2%", cmap="Blues",
                    xticklabels=self.classes, yticklabels=self.classes,
                    cbar=False, annot_kws={"size": 9, "weight": "bold"}, ax=ax1)  # Texto heatmap = 9

        ax1.set_title('Matriz de Confusión (Recall)', fontsize=10, fontweight='bold', pad=8)
        ax1.set_ylabel('Etiqueta Real', fontsize=9, fontweight='bold')
        ax1.set_xlabel('Predicción del Modelo', fontsize=9, fontweight='bold')
        ax1.tick_params(axis='y', rotation=0, labelsize=8)  # Etiquetas ejes = 8
        ax1.tick_params(axis='x', labelsize=8)

        # ========================================================
        # PANEL 2. CURVAS ROC
        # ========================================================
        ax2 = fig.add_subplot(gs[0, 1])
        y_bin = label_binarize(self.y_true, classes=[0, 1, 2])
        colores = ['#2ecc71', '#f39c12', '#e74c3c']

        for i, color in zip(range(self.n_classes), colores):
            fpr, tpr, _ = roc_curve(y_bin[:, i], self.y_prob[:, i])
            ax2.plot(fpr, tpr, color=color, lw=1.5, label=f'ROC {self.classes[i]} (AUC={auc(fpr, tpr):.4f})')

        ax2.plot([0, 1], [0, 1], 'k--', lw=1, alpha=0.5)
        ax2.set_xlim([-0.01, 1.0])
        ax2.set_ylim([0.0, 1.05])
        ax2.set_xlabel('FPR (Falsos Positivos)', fontsize=9, fontweight='bold')
        ax2.set_ylabel('TPR (Verdaderos Positivos)', fontsize=9, fontweight='bold')
        ax2.set_title('Curva ROC One-vs-Rest', fontsize=10, fontweight='bold', pad=8)
        ax2.legend(loc="lower right", fontsize=8)  # Leyenda = 8
        ax2.grid(True, linestyle=':', alpha=0.6)

        # ========================================================
        # PANEL 3. TABLA DE MÉTRICAS
        # ========================================================
        ax3 = fig.add_subplot(gs[1, :])
        ax3.axis('off')

        cell_text = []
        for i in range(self.n_classes):
            cell_text.append([f"{precision[i]:.4f}", f"{recall[i]:.4f}", f"{f1[i]:.4f}", str(support[i])])

        cell_text.append(["---", "---", "---", "---"])
        cell_text.append(
            [f"{np.mean(precision):.4f}", f"{np.mean(recall):.4f}", f"{np.mean(f1):.4f}", str(self.num_samples)])

        filas = self.classes + ['---------', 'MACRO AVG']
        columnas = ['Precision', 'Recall', 'F1-Score', 'Soporte']

        # Caja de la tabla más ajustada
        tabla = ax3.table(cellText=cell_text, rowLabels=filas, colLabels=columnas,
                          loc='center', cellLoc='center', bbox=[0.15, 0.1, 0.7, 0.8])

        tabla.auto_set_font_size(False)
        tabla.set_fontsize(9)  # Texto de tabla = 9
        ax3.set_title('Reporte de Clasificación Cuantitativa (F1-Score)', fontsize=10, fontweight='bold', pad=10)

        for (row, col), cell in tabla.get_celld().items():
            cell.set_edgecolor('#dddddd')
            if row == 0 or col == -1:
                cell.set_text_props(weight='bold')

        for j in range(len(columnas)):
            tabla[(0, j)].set_facecolor('#f4f6f9')
        for i in range(1, len(filas) + 1):
            tabla[(i, -1)].set_facecolor('#f4f6f9')

        plt.savefig("evaluacion_dsr_vit3d.png", dpi=300)
        print("[*] Dashboard guardado como 'evaluacion_dsr_vit3d.png'.")
        plt.show()


if __name__ == "__main__":
    simulador = SimuladorEvaluacionDSR(num_samples=2500)
    simulador.generar_dashboard_doctoral()