import torch
import cv2
from models.anomaly_detector import AnomalyDetector
from data.transforms import get_val_transforms


class ViTPredictor:
    """
    Carga el Vision Transformer ya entrenado y procesa fotogramas en tiempo real.
    """

    def __init__(self, model_path, device=None):
        # 1. Configurar dispositivo (GPU si está disponible)
        self.device = device if device else torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"[*] Motor de inferencia inicializado en: {self.device.type.upper()}")

        # Diccionario de clases según cómo entrenamos el modelo
        self.classes = {0: 'NORMAL', 1: 'VIOLENCIA DETECTADA', 2: 'ARMA DETECTADA'}

        # 2. Inicializar la arquitectura vacía y cargarle el "cerebro" (.pth)
        self.model = AnomalyDetector(num_classes=3).to(self.device)
        self.model.load_state_dict(torch.load(model_path, map_location=self.device, weights_only=True))

        # Congelar el modelo para que no aprenda más, solo prediga (ahorra muchísima memoria)
        self.model.eval()

        # 3. Cargar las transformaciones de validación (sin giros ni alteraciones aleatorias)
        self.transform = get_val_transforms(img_size=224)

    def predict_frame(self, frame):
        """
        Recibe un frame de OpenCV (BGR) y devuelve la clase detectada y su nivel de confianza.
        """
        # Convertir a RGB (formato con el que entrenó la IA)
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Transformar a tensor y añadir dimensión de batch (1, C, H, W)
        tensor = self.transform(rgb_frame).unsqueeze(0).to(self.device)

        # Desactivar el cálculo de gradientes para máxima velocidad
        with torch.no_grad():
            outputs = self.model(tensor)

            # Convertir la salida bruta en porcentajes de probabilidad (0.0 a 1.0)
            probabilities = torch.softmax(outputs, dim=1)[0]

            # Extraer la opción con el porcentaje más alto
            confidence, predicted_idx = torch.max(probabilities, 0)

        return self.classes[predicted_idx.item()], confidence.item()


# ==========================================
# Prueba de la Inferencia (Se ejecutará cuando termine el entrenamiento)
# ==========================================
if __name__ == "__main__":
    import os
    import numpy as np

    # Buscar el modelo de la época 10
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    ruta_modelo = os.path.join(base_dir, "models", "vit_checkpoint_epoch_10.pth")

    if os.path.exists(ruta_modelo):
        # Cargar el predictor
        detector = ViTPredictor(model_path=ruta_modelo)

        # Crear un fotograma de ruido falso para probar la latencia
        frame_prueba = np.random.randint(0, 255, (720, 1280, 3), dtype=np.uint8)

        etiqueta, confianza = detector.predict_frame(frame_prueba)
        print(f"\nResultado del test:")
        print(f"Clasificación: {etiqueta}")
        print(f"Confianza:     {confianza:.2%}")
    else:
        print(f"Esperando a que termine el entrenamiento...\nNo se encontró el archivo: {ruta_modelo}")