import torch
import cv2
import numpy as np
from collections import deque
from models.spatiotemporal_vit import SpatioTemporalViT
from data.transforms import get_val_transforms


class ViTPredictor3D:
    """
    Motor de inferencia doctoral con Fusión Híbrida (Deep Learning + MOG2).
    Utiliza Modelado de Fondo Gaussiano para crear una máscara física impenetrable,
    aniquilando logos estáticos y forzando a la IA a auditar solo sujetos vivos.
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
        self.block_features = None

        # --- INNOVACIÓN FINAL: MOG2 (Mixture of Gaussians) ---
        # Aprende el fondo del CCTV de forma inteligente, ignorando logos estáticos.
        self.bg_subtractor = cv2.createBackgroundSubtractorMOG2(history=500, varThreshold=25, detectShadows=False)

        last_block = self.model.spatial_backbone.blocks[-1]
        last_block.register_forward_hook(self._hook_fn)

    def _hook_fn(self, module, input, output):
        self.block_features = output.detach().cpu()

    def predict_frame(self, frame):
        # 1. Alimentar el modelo MOG2 para que aprenda el escenario en tiempo real
        fg_mask = self.bg_subtractor.apply(frame)

        # 2. Preparar tensor para el Transformer
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        tensor = self.transform(rgb_frame)
        self.frame_buffer.append(tensor)

        if len(self.frame_buffer) < self.seq_length:
            return "ANALIZANDO CINEMÁTICA...", 0.0, None

        seq_tensor = torch.stack(list(self.frame_buffer)).unsqueeze(0).to(self.device)

        with torch.no_grad():
            outputs = self.model(seq_tensor)
            probabilities = torch.softmax(outputs, dim=1)[0]
            confidence, predicted_idx = torch.max(probabilities, 0)

        heatmap_ui = None
        if self.block_features is not None:
            # ========================================================
            # ESCUDO FÍSICO IMPENETRABLE (MOG2 + MORFOLOGÍA)
            # ========================================================
            # Limpiamos micro-ruidos (compresión) y engordamos las siluetas humanas
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
            fg_mask_clean = cv2.morphologyEx(fg_mask, cv2.MORPH_OPEN, kernel)
            fg_mask_clean = cv2.dilate(fg_mask_clean, kernel, iterations=4)

            # Convertimos la silueta a un multiplicador suave (0.0 a 1.0)
            physical_mask = fg_mask_clean.astype(np.float32) / 255.0
            physical_mask = cv2.resize(physical_mask, (640, 480))
            physical_mask = cv2.GaussianBlur(physical_mask, (31, 31), 0)

            # ========================================================
            # CAPA NEURONAL (Varianza ViT)
            # ========================================================
            patch_features = self.block_features[:, 1:, :]
            variance_over_time = patch_features.var(dim=0)
            motion_map = variance_over_time.sum(dim=-1)

            mapa_2d = motion_map.view(14, 14).numpy()

            min_val, max_val = mapa_2d.min(), mapa_2d.max()
            if max_val > min_val:
                mapa_2d = (mapa_2d - min_val) / (max_val - min_val)
            else:
                mapa_2d = np.zeros_like(mapa_2d)

            mapa_2d = np.power(mapa_2d, 1.5)
            neural_map = cv2.resize(mapa_2d, (640, 480), interpolation=cv2.INTER_CUBIC)

            # ========================================================
            # FUSIÓN DEFINITIVA (Saliency Gating Híbrido)
            # ========================================================
            # La física bloquea implacablemente cualquier zona donde no haya una persona viva moviéndose.
            fused_map = neural_map * physical_mask

            # Normalizamos el remanente para que brille intensamente sobre los agresores
            if fused_map.max() > 0.02:
                fused_map = fused_map / fused_map.max()

            fused_map = np.uint8(255 * fused_map)
            heatmap_ui = cv2.applyColorMap(fused_map, cv2.COLORMAP_JET)

        return self.classes[predicted_idx.item()], confidence.item(), heatmap_ui