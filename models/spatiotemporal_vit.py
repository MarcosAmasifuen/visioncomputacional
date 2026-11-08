import torch
import torch.nn as nn
import timm


class SpatioTemporalViT(nn.Module):
    def __init__(self, num_classes=3, num_frames=5, embed_dim=192, num_heads=4):
        super().__init__()
        self.num_frames = num_frames

        self.spatial_backbone = timm.create_model('vit_tiny_patch16_224', pretrained=True)
        self.spatial_backbone.head = nn.Identity()

        # INNOVACIÓN DOCTORAL: Fine-tuning del último bloque.
        # En lugar de congelar todo, dejamos libre el último bloque (blocks.11)
        # y la capa de normalización para que se adapten a las cámaras CCTV.
        for name, param in self.spatial_backbone.named_parameters():
            if 'blocks.11' not in name and 'norm' not in name:
                param.requires_grad = False
            else:
                param.requires_grad = True  # Descongelamos la última capa

        self.temporal_pos_embed = nn.Parameter(torch.zeros(1, num_frames, embed_dim))

        encoder_layer = nn.TransformerEncoderLayer(d_model=embed_dim, nhead=num_heads, batch_first=True)
        self.temporal_transformer = nn.TransformerEncoder(encoder_layer, num_layers=2)

        self.classifier = nn.Sequential(
            nn.Linear(embed_dim, 128),
            nn.GELU(),
            nn.Dropout(0.4),
            nn.Linear(128, num_classes)
        )

    def forward(self, x):
        B, T, C, H, W = x.shape
        x = x.view(B * T, C, H, W)

        spatial_features = self.spatial_backbone(x)
        temporal_sequence = spatial_features.view(B, T, -1)

        temporal_sequence = temporal_sequence + self.temporal_pos_embed
        contextualized_sequence = self.temporal_transformer(temporal_sequence)

        sequence_summary = contextualized_sequence[:, -1, :]
        return self.classifier(sequence_summary)