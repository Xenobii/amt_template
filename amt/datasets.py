import os
import csv
from typing import List
from torch.utils.data import Dataset



class MaestroDataset(Dataset):
    def __init__(self, root: str, split: str):
        super().__init__()

        csv_path = os.path.join(root, "maestro-v3.0.0.csv")
        self.wav_files: List[str] = []
        self.mid_files: List[str] = []

        with open(csv_path, newline='', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            for row in reader:
                if row['split'] == split:
                    self.wav_files.append(os.path.join(root, row['audio_filename']))
                    self.mid_files.append(os.path.join(root, row['midi_filename']))

    def __len__(self) -> int:
        return len(self.wav_files)

    def __getitem__(self, index: int) -> str:
        return {
            "wav": self.wav_files[index],
            "mid": self.mid_files[index],
        }