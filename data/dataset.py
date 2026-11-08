import os
import cv2
import torch
from torch.utils.data import Dataset
from data.transforms import get_train_transforms, get_val_transforms


class CCTVFrameDataset(Dataset):
    def __init__(self, root_dir, is_train=True, img_size=224):
        self.root_dir = root_dir
        self.transform = get_train_transforms(img_size) if is_train else get_val_transforms(img_size)

        self.filepaths = []
        self.labels = []

        # Mapeo de las carpetas de Kaggle a índices numéricos
        classes = {'Normal': 0, 'Violence': 1, 'Weaponized': 2}

        # Escanear los directorios y armar el índice
        for class_name, label in classes.items():
            class_dir = os.path.join(root_dir, class_name)
            if os.path.exists(class_dir):
                for fname in os.listdir(class_dir):
                    if fname.lower().endswith(('.png', '.jpg', '.jpeg')):
                        self.filepaths.append(os.path.join(class_dir, fname))
                        self.labels.append(label)

    def __len__(self):
        return len(self.filepaths)

    def __getitem__(self, idx):
        img_path = self.filepaths[idx]
        label = self.labels[idx]

        img = cv2.imread(img_path)
        if img is None:
            return torch.zeros(3, 224, 224), torch.tensor(label, dtype=torch.long)

        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        if self.transform:
            img = self.transform(img)

        return img, torch.tensor(label, dtype=torch.long)


if __name__ == "__main__":
    # Autodetectar la ruta de la carpeta processed
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    ruta_processed = os.path.join(base_dir, "data", "processed")

    dataset = CCTVFrameDataset(root_dir=ruta_processed, is_train=True)
    print(f"Total de imágenes listas para entrenar: {len(dataset)}")