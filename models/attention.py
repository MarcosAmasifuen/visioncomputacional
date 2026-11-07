import torch
import torch.nn as nn


class MultiHeadSelfAttention(nn.Module):
    """
    Mecanismo de atención para encontrar relaciones espaciales entre todos los parches de la imagen CCTV.
    """

    def __init__(self, embed_dim=768, num_heads=12, dropout=0.0):
        super().__init__()
        assert embed_dim % num_heads == 0, "El embed_dim debe ser divisible por num_heads"

        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads

        # Factor de escalado para evitar que los gradientes exploten
        self.scale = self.head_dim ** -0.5

        # Generamos Query (Q), Key (K) y Value (V) en una sola operación lineal por eficiencia en GPU
        self.qkv = nn.Linear(embed_dim, embed_dim * 3, bias=False)
        self.attn_drop = nn.Dropout(dropout)

        # Proyección de salida
        self.proj = nn.Linear(embed_dim, embed_dim)
        self.proj_drop = nn.Dropout(dropout)

    def forward(self, x, return_attention=False):
        """
        Entrada x: Secuencia de parches -> (Batch, Num_Patches, Embed_Dim)
        """
        B, N, C = x.shape

        # 1. Proyectar Q, K, V y separarlos en múltiples "cabezales" de atención
        # Forma resultante: (3, Batch, Num_Heads, Num_Patches, Head_Dim)
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]

        # 2. Calcular los puntajes de atención espacial (Q multiplicado por K transpuesto)
        # Forma: (Batch, Num_Heads, Num_Patches, Num_Patches)
        attn = (q @ k.transpose(-2, -1)) * self.scale

        # Normalizar a probabilidades entre 0 y 1
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        # 3. Aplicar la atención sobre los Valores (V) y recombinar los cabezales
        x = (attn @ v).transpose(1, 2).reshape(B, N, C)

        # 4. Capa lineal final
        x = self.proj(x)
        x = self.proj_drop(x)

        if return_attention:
            # Retornamos los pesos para generar el mapa de calor (Heatmap) en producción
            return x, attn

        return x


# ==========================================
# Prueba rápida de dimensiones
# ==========================================
if __name__ == "__main__":
    # Simulamos la salida del archivo anterior: 2 videos, 196 parches, 768 de dimensión
    dummy_patches = torch.randn(2, 196, 768)

    attention_module = MultiHeadSelfAttention()

    # Procesamos simulando que estamos en modo inferencia para extraer el mapa
    output, attention_map = attention_module(dummy_patches, return_attention=True)

    print(f"Entrada (Parches procesados): {dummy_patches.shape}")
    print(f"Salida (Parches con contexto):{output.shape}")
    print(f"Mapa de Atención (Heatmap):   {attention_map.shape} -> (Batch, Heads, Patches, Patches)")