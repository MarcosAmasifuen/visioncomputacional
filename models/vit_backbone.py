import torch
import torch.nn as nn
from models.patch_embedding import PatchEmbedding
from models.attention import MultiHeadSelfAttention


class MLP(nn.Module):
    """Red neuronal de dos capas que procesa los tokens después de la atención."""

    def __init__(self, in_features, hidden_features, out_features, drop=0.0):
        super().__init__()
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = nn.GELU()  # Función de activación más estable que ReLU para Transformers
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class TransformerBlock(nn.Module):
    """Un bloque completo del Vision Transformer: Norm -> Atención -> Norm -> MLP"""

    def __init__(self, embed_dim=768, num_heads=12, mlp_ratio=4.0, drop=0.0):
        super().__init__()
        self.norm1 = nn.LayerNorm(embed_dim)
        self.attn = MultiHeadSelfAttention(embed_dim, num_heads, drop)

        self.norm2 = nn.LayerNorm(embed_dim)
        hidden_features = int(embed_dim * mlp_ratio)
        self.mlp = MLP(embed_dim, hidden_features, embed_dim, drop)

    def forward(self, x, return_attention=False):
        # Conexión residual 1 (Suma la entrada original a la salida de la atención)
        if return_attention:
            attn_out, attn_map = self.attn(self.norm1(x), return_attention=True)
            x = x + attn_out
        else:
            x = x + self.attn(self.norm1(x))
            attn_map = None

        # Conexión residual 2 (Suma a la salida del MLP)
        x = x + self.mlp(self.norm2(x))

        if return_attention:
            return x, attn_map
        return x


class ViTBackbone(nn.Module):
    """Ensambla todo el modelo Vision Transformer base para procesar frames estáticos."""

    def __init__(self, img_size=224, patch_size=16, in_channels=3, embed_dim=768, depth=12, num_heads=12):
        super().__init__()
        self.patch_embed = PatchEmbedding(img_size, patch_size, in_channels, embed_dim)

        # Token de Clasificación [CLS] y Embeddings de Posición
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        # Sumamos +1 a los parches para incluir el token [CLS]
        self.pos_embed = nn.Parameter(torch.zeros(1, self.patch_embed.num_patches + 1, embed_dim))
        self.pos_drop = nn.Dropout(p=0.0)

        # Apilamos múltiples bloques Transformer (depth = 12 por defecto)
        self.blocks = nn.ModuleList([
            TransformerBlock(embed_dim, num_heads) for _ in range(depth)
        ])
        self.norm = nn.LayerNorm(embed_dim)

    def forward(self, x, return_attention=False):
        B = x.shape[0]
        # 1. Extraer parches
        x = self.patch_embed(x)

        # 2. Añadir el token [CLS] al inicio de la secuencia
        cls_tokens = self.cls_token.expand(B, -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)

        # 3. Sumar la información espacial (posición)
        x = x + self.pos_embed
        x = self.pos_drop(x)

        # 4. Procesar a través de todos los bloques Transformer
        attn_maps = []
        for block in self.blocks:
            if return_attention:
                x, attn_map = block(x, return_attention=True)
                attn_maps.append(attn_map)
            else:
                x = block(x)

        x = self.norm(x)

        if return_attention:
            return x, attn_maps
        return x


# ==========================================
# Prueba de la Arquitectura Base Completa
# ==========================================
if __name__ == "__main__":
    # Creamos un batch de 2 imágenes de 224x224
    dummy_cctv_frame = torch.randn(2, 3, 224, 224)

    # Instanciamos un ViT pequeño (4 capas de profundidad para probar rápido en local)
    vit_model = ViTBackbone(depth=4)

    output, attention_maps = vit_model(dummy_cctv_frame, return_attention=True)

    print(f"Frame de entrada:        {dummy_cctv_frame.shape}")
    print(f"Salida del ViT Backbone: {output.shape} -> (Batch, Parches + CLS, Embed_Dim)")
    print(f"Capas de atención ext.:  {len(attention_maps)} (Una por bloque Transformer)")
    print(f"Shape del mapa capa 1:   {attention_maps[0].shape}")