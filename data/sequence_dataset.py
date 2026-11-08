import os
import cv2
import torch
from torch.utils.data import Dataset
from collections import defaultdict
from data.transforms import get_train_transforms, get_val_transforms


class CCTVSequenceDataset(Dataset):
    def __init__(self, root_dir, is_train=True, img_size=224, seq_length=5):
        self.root_dir = root_dir
        self.transform = get_train_transforms(img_size) if is_train else get_val_transforms(img_size)
        self.seq_length = seq_length
        self.sequences = []
        self.labels = []

        classes = {'Normal': 0, 'Violence': 1, 'Weaponized': 2}

        for class_name, label in classes.items():
            class_dir = os.path.join(root_dir, class_name)
            if not os.path.exists(class_dir):
                continue

            video_frames = defaultdict(list)
            for fname in os.listdir(class_dir):
                if fname.lower().endswith('.jpg'):
                    vid_id = fname.split('_frame')[0]
                    video_frames[vid_id].append(fname)

            for vid_id, frames in video_frames.items():
                frames.sort(key=lambda x: int(x.split('_frame')[1].split('.jpg')[0]))

                if len(frames) >= self.seq_length:
                    # INNOVACIÓN DOCTORAL: Ventana Deslizante (Sliding Window) con paso=1
                    # Esto extrae secuencias superpuestas, multiplicando el dataset masivamente
                    for i in range(len(frames) - self.seq_length + 1):
                        seq_paths = [os.path.join(class_dir, f) for f in frames[i:i + self.seq_length]]
                        self.sequences.append(seq_paths)
                        self.labels.append(label)

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, idx):
        seq_paths = self.sequences[idx]
        label = self.labels[idx]

        seq_tensors = []
        for img_path in seq_paths:
            img = cv2.imread(img_path)
            if img is None:
                img_tensor = torch.zeros(3, 224, 224)
            else:
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                img_tensor = self.transform(img)
            seq_tensors.append(img_tensor)

        return torch.stack(seq_tensors), torch.tensor(label, dtype=torch.long)


if __name__ == "__main__":
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    ruta_processed = os.path.join(base_dir, "data", "processed")
    seq_dataset = CCTVSequenceDataset(root_dir=ruta_processed, is_train=True, seq_length=5)
    print(f"Total de secuencias tras Ventana Deslizante: {len(seq_dataset)}")