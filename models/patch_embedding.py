import torch
import torch.nn as nn


class PatchEmbedding(nn.Module):
    """
    Divide una imagen en parches y los proyecta a un espacio vectorial (Embeddings).
    """

    def __init__(self, img_size=224, patch_size=16, in_channels=3, embed_dim=768):
        super().__init__()
        self.img_size = img_size
        self.patch_size = patch_size

        # Calculamos cuántos parches se generarán (ej. 224/16 = 14 -> 14x14 = 196 parches)
        self.num_patches = (img_size // patch_size) ** 2

        # Usamos un filtro convolucional del mismo tamaño que el parche
        # y con un "salto" (stride) igual al tamaño del parche para que no se superpongan.
        self.proj = nn.Conv2d(
            in_channels,
            embed_dim,
            kernel_size=patch_size,
            stride=patch_size
        )

    def forward(self, x):
        """
        Entrada x: Tensor de imágenes CCTV -> (Batch, Canales, Alto, Ancho)
        Salida: Secuencia de tokens de parches -> (Batch, Num_Patches, Embed_Dim)
        """
        # 1. Extraer los parches
        # (Batch, 3, 224, 224) -> (Batch, 768, 14, 14)
        x = self.proj(x)

        # 2. Aplanar el mapa 2D a una secuencia 1D de parches
        # (Batch, 768, 14, 14) -> (Batch, 768, 196)
        x = x.flatten(2)

        # 3. Transponer para que la dimensión del vector quede al final
        # (Batch, 768, 196) -> (Batch, 196, 768)
        x = x.transpose(1, 2)

        return x


# ==========================================
# Prueba rápida (puedes ejecutar este archivo)
# ==========================================
if __name__ == "__main__":
    # Simulamos un "batch" de 2 imágenes de cámara de seguridad a color (3 canales) de 224x224
    dummy_cctv_frame = torch.randn(2, 3, 224, 224)

    patch_embedder = PatchEmbedding()
    output = patch_embedder(dummy_cctv_frame)

    print(f"Dimensiones de entrada: {dummy_cctv_frame.shape}")
    print(f"Dimensiones de salida:  {output.shape} -> (Batch, Num_Patches, Embed_Dim)")