import os
import h5py
import numpy as np
import torch
from tqdm import tqdm
from omegaconf import DictConfig
from torch.utils.data import Dataset 
from hydra.utils import instantiate
from typing import Tuple



class Corpus(Dataset):
    def __init__(self, corpus_file: str, split=None, **kwargs):
        super().__init__()

        self.h5_path   = corpus_file
        self.split     = split

        with h5py.File(self.h5_path, "r") as h5:
            self.keys = sorted(list(h5[split].keys()))

        self._h5 = None

    def _get_h5(self):
        if self._h5 is None:
            self._h5 = h5py.File(self.h5_path, "r")
        return self._h5

    def __len__(self):
        return len(self.keys)

    def __getitem__(self, idx: int) -> Tuple:
        h5 = self._get_h5()
        grp = h5[self.split][self.keys[idx]]

        out = {}
        # data
        for k in grp.keys():
            v = grp[k][()]
            out[k] = torch.from_numpy(v, dtype=torch.float32)
        # attrs
        for name, val in grp.attrs.items():
            out[name] = val

        return out

def create_corpus(cfg: DictConfig):
    # --- save dir ---
    f_out = cfg.corpus.corpus_file
    out_dir = os.path.dirname(f_out)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)

    # --- dataset ---
    train_dataset = instantiate(cfg.dataset, split="train")
    valid_dataset = instantiate(cfg.dataset, split="validation")

    # --- model ---
    model = instantiate(cfg.model)

    with h5py.File(f_out, "w") as h5:
        print(f"Creating corpus...")

        train_data = h5.create_group("train")
        valid_data = h5.create_group("valid")

        train_file_indices = np.arange(len(train_dataset))
        valid_file_indices = np.arange(len(valid_dataset))

        print(f"Instantiated split: train_len={len(train_dataset)}, valid_len={len(valid_dataset)}")
        
        # --- train data group ---
        for idx in tqdm(train_file_indices, desc="Creating training corpus"):
            # get files
            files = train_dataset[idx]
            wav_file = files["wav"]
            mid_file = files["mid"]
            
            # process inputs
            output = model.process_input(wav_file, mid_file)
            
            # create a group per chunk
            n_chunks = output["spec"].shape[0]
            for i in range(n_chunks):
                group = train_data.create_group(f"{idx:07}_{i:04d}")
                for k, v in output.items():
                    group.create_dataset(k, data=v[i].cpu().numpy(), compression="lzf")
                
                group.attrs["wav_file"] = wav_file
                group.attrs["mid_file"] = mid_file
        
        # --- valid data group ---
        for idx in tqdm(valid_file_indices, desc="Creating training corpus"):
            # get files
            files = valid_dataset[idx]
            wav_file = files["wav"]
            mid_file = files["mid"]
            
            # process inputs
            output = model.process_input(wav_file, mid_file)
            
            # create a group per chunk
            n_chunks = output["spec"].shape[0]
            for i in range(n_chunks):
                group = valid_data.create_group(f"{idx:07}_{i:04d}")
                for k, v in output.items():
                    group.create_dataset(k, data=v[i].cpu().numpy(), compression="lzf")
                
                group.attrs["wav_file"] = wav_file
                group.attrs["mid_file"] = mid_file

    print(f"Finished processing. Dataset saved at {f_out}")