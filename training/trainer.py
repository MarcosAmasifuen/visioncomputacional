import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader
from models.anomaly_detector import AnomalyDetector
from data.dataset import CCTVFrameDataset
import time
import os


def train_model(data_dir, num_epochs=5, batch_size=16, learning_rate=5e-5):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\nIniciando ENTRENAMIENTO AVANZADO en: {device.type.upper()}")
    print("-" * 40)

    train_dataset = CCTVFrameDataset(root_dir=data_dir, is_train=True)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0)

    # Inicializar Modelo (Descargará los pesos de timm la primera vez)
    model = AnomalyDetector(num_classes=3).to(device)

    # Hiperparámetros Optimizados
    criterion = nn.CrossEntropyLoss()

    # Optimizador con mayor Weight Decay (1e-3)
    optimizer = optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=1e-3)

    # Planificador que ajusta el LR en forma de curva cosenoidal
    scheduler = CosineAnnealingLR(optimizer, T_max=num_epochs)

    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        correct_predictions = 0
        total_samples = 0

        start_time = time.time()

        for batch_idx, (images, labels) in enumerate(train_loader):
            images, labels = images.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)

            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            _, predicted = torch.max(outputs, 1)
            correct_predictions += (predicted == labels).sum().item()
            total_samples += labels.size(0)

            if batch_idx % 20 == 0:
                print(
                    f"Epoch [{epoch + 1}/{num_epochs}] Lote [{batch_idx}/{len(train_loader)}] | Pérdida: {loss.item():.4f}")

        # Actualizar el Scheduler al final de cada Epoch
        scheduler.step()

        epoch_loss = running_loss / len(train_loader)
        epoch_acc = (correct_predictions / total_samples) * 100
        epoch_time = time.time() - start_time

        # Extraer el Learning Rate actual usado en este Epoch
        current_lr = scheduler.get_last_lr()[0]

        print(f"\n[!] Fin de Epoch {epoch + 1}")
        print(
            f"Precisión: {epoch_acc:.2f}% | Pérdida: {epoch_loss:.4f} | LR: {current_lr:.6f} | Tiempo: {epoch_time:.1f}s\n")
        print("-" * 40)

        base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        save_path = os.path.join(base_dir, "models", f"vit_checkpoint_epoch_{epoch + 1}.pth")
        torch.save(model.state_dict(), save_path)


if __name__ == "__main__":
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    ruta_processed = os.path.join(base_dir, "data", "processed")
    ruta_models = os.path.join(base_dir, "models")

    os.makedirs(ruta_models, exist_ok=True)

    # Lanzamos el entrenamiento con 5 Epochs (suficiente para Transfer Learning)
    train_model(data_dir=ruta_processed, num_epochs=5, batch_size=16)