import torch
import cv2
import numpy as np
from collections import deque
from models.spatiotemporal_vit import SpatioTemporalViT
from data.transforms import get_val_transforms


class ViTPredictor3D:
    """
    Motor de inferencia doctoral con Explicabilidad XAI Grad-CAM Dinámico.
    Garantiza el realce visual adaptativo en cualquier fotograma.
    """

    def __init__(self, model_path, seq_length=5, device=None):
        self.device = device if device else torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.seq_length = seq_length
        self.classes = {0: 'NORMAL', 1: 'VIOLENCIA DETECTADA', 2: 'ARMA DETECTADA'}

        self.model = SpatioTemporalViT(num_classes=3, num_frames=seq_length).to(self.device)
        self.model.load_state_dict(torch.load(model_path, map_location=self.device, weights_only=True))
        self.model.eval()

        self.transform = get_val_transforms(img_size=224)
        self.frame_buffer = deque(maxlen=seq_length)

        self.activations = None
        self.gradients = None

        target_layer = self.model.spatial_backbone.norm
        target_layer.register_forward_hook(self._forward_hook)
        target_layer.register_full_backward_hook(self._backward_hook)

    def _forward_hook(self, module, input, output):
        self.activations = output

    def _backward_hook(self, module, grad_input, grad_output):
        self.gradients = grad_output[0]

    def predict_frame(self, frame):
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        tensor = self.transform(rgb_frame)
        self.frame_buffer.append(tensor)

        if len(self.frame_buffer) < self.seq_length:
            return "ANALIZANDO CINEMÁTICA...", 0.0, None

        seq_tensor = torch.stack(list(self.frame_buffer)).unsqueeze(0).to(self.device)

        with torch.enable_grad():
            seq_tensor.requires_grad_(True)
            self.model.zero_grad()

            outputs = self.model(seq_tensor)
            probabilities = torch.softmax(outputs, dim=1)[0]
            confidence, predicted_idx = torch.max(probabilities, 0)

            if confidence > 0:
                outputs[0, predicted_idx].backward()

        heatmap_ui = None
        if self.activations is not None and self.gradients is not None:
            acts = self.activations[:, 1:, :]
            grads = self.gradients[:, 1:, :]

            weights = grads.mean(dim=1, keepdim=True)
            cam = (weights * acts).sum(dim=-1)
            cam = torch.relu(cam)

            cam_promedio = cam.mean(dim=0).detach().cpu().numpy()
            mapa_2d = cam_promedio.reshape(14, 14)

            # --- ESCALAMIENTO DINÁMICO ADAPTATIVO ---
            min_val, max_val = mapa_2d.min(), mapa_2d.max()
            if max_val > min_val:
                mapa_2d = (mapa_2d - min_val) / (max_val - min_val)
            else:
                mapa_2d = np.zeros_like(mapa_2d)

            # Realce no lineal (Potencia) para aumentar el contraste de los puntos clave
            mapa_2d = np.power(mapa_2d, 0.7)

            # Convertir a imagen térmica nítida
            mapa_2d = np.uint8(255 * mapa_2d)
            heatmap_ui = cv2.resize(mapa_2d, (640, 480), interpolation=cv2.INTER_CUBIC)
            heatmap_ui = cv2.applyColorMap(heatmap_ui, cv2.COLORMAP_JET)

        return self.classes[predicted_idx.item()], confidence.item(), heatmap_ui