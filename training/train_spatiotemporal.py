import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from models.spatiotemporal_vit import SpatioTemporalViT
from data.sequence_dataset import CCTVSequenceDataset
import time
import os


def train_temporal_model(data_dir, num_epochs=8, batch_size=4, seq_length=5, learning_rate=1e-4):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\n[ EXPERIMENTO DOCTORAL: MODELADO ESPACIO-TEMPORAL ]")
    print(f"Dispositivo: {device.type.upper()} | Frames por secuencia: {seq_length}")
    print("-" * 50)

    # 1. Cargar el dataset de secuencias (Tuberías de tiempo)
    train_dataset = CCTVSequenceDataset(root_dir=data_dir, is_train=True, seq_length=seq_length)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0)

    # 2. Inicializar el ViT Espacio-Temporal
    model = SpatioTemporalViT(num_classes=3, num_frames=seq_length).to(device)

    # 3. Solo entrenamos el Transformador Temporal y el Clasificador
    # (El backbone espacial de timm lo congelamos en la arquitectura para ahorrar memoria)
    parametros_entrenables = [p for p in model.parameters() if p.requires_grad]

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(parametros_entrenables, lr=learning_rate, weight_decay=1e-3)

    # 4. Bucle de Entrenamiento
    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        correct_predictions = 0
        total_samples = 0

        start_time = time.time()

        for batch_idx, (secuencias, labels) in enumerate(train_loader):
            # secuencias shape: [Batch, Time(5), Channels(3), Height(224), Width(224)]
            secuencias, labels = secuencias.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(secuencias)
            loss = criterion(outputs, labels)

            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            _, predicted = torch.max(outputs, 1)
            correct_predictions += (predicted == labels).sum().item()
            total_samples += labels.size(0)

            if batch_idx % 10 == 0:
                print(
                    f"Epoch [{epoch + 1}/{num_epochs}] Lote [{batch_idx}/{len(train_loader)}] | Pérdida: {loss.item():.4f}")

        epoch_loss = running_loss / len(train_loader)
        epoch_acc = (correct_predictions / total_samples) * 100
        epoch_time = time.time() - start_time

        print(f"\n[!] Fin de Epoch {epoch + 1}")
        print(f"Precisión Cinematica: {epoch_acc:.2f}% | Pérdida: {epoch_loss:.4f} | Tiempo: {epoch_time:.1f}s\n")
        print("-" * 50)

        # Guardar el artefacto de investigación
        base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        save_path = os.path.join(base_dir, "models", f"spatiotemporal_vit_epoch_{epoch + 1}.pth")
        torch.save(model.state_dict(), save_path)


if __name__ == "__main__":
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    ruta_processed = os.path.join(base_dir, "data", "processed")
    ruta_models = os.path.join(base_dir, "models")

    os.makedirs(ruta_models, exist_ok=True)

    # Arrancamos el experimento con secuencias de 5 fotogramas
    train_temporal_model(data_dir=ruta_processed, num_epochs=8, batch_size=4, seq_length=5)