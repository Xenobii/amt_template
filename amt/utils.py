import matplotlib.pyplot as plt
import numpy as np
import torch
import h5py


def plot_spec(*args, k=0, cmap="magma", save_path=None):
    """
    Plot one or more spectrograms in the same figure.
    Usage:
      plot_spec(spec)                   # single spec, default k=0
      plot_spec(spec, 2)                # single spec, k=2 (backwards-compatible)
      plot_spec(spec1, spec2, k=0)      # multiple specs
    Each spec must have shape [B, F, T] (batch/channel, frequency, time).
    """
    specs = list(args)
    if len(specs) == 0:
        raise ValueError("No spectrograms provided")

    if isinstance(specs[-1], (int, np.integer)):
        k = int(specs.pop())

    if len(specs) == 0:
        raise ValueError("No spectrograms provided")

    mats = []
    for idx, spec in enumerate(specs):
        if isinstance(spec, torch.Tensor):
            arr = spec.detach().cpu().numpy()
        else:
            arr = np.asarray(spec)

        if arr.ndim != 3:
            raise ValueError(f"spec {idx} must have 3 dims [B,F,T], got {arr.ndim}")

        B, F, T = arr.shape
        if not (0 <= k < B):
            raise IndexError(f"k out of range for spec {idx}: got {k}, expected 0 <= k < {B}")

        mat = arr[k].T  # convert [F, T] -> [T, F] to preserve original plotting path
        if np.iscomplexobj(mat):
            mat = np.abs(mat)

        mats.append(mat)

    n = len(mats)
    fig, axes = plt.subplots(1, n, figsize=(6 * n, 3))
    if n == 1:
        axes = [axes]

    for i, ax in enumerate(axes):
        im = ax.imshow(mats[i].T, aspect="auto", origin="lower", cmap=cmap)
        ax.set_xlabel("Time")
        ax.set_ylabel("Frequency")
        ax.set_title(f"Spectrogram [{k}] ({i})")
        plt.colorbar(im, ax=ax, label="Amplitude")

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path)
        plt.close()
    else:
        plt.show()


    
def print_h5_structure(h5_path):
    with h5py.File(h5_path, "r") as f:
        def _print(name, obj):
            depth = name.count("/")
            indent = "  " * depth
            if isinstance(obj, h5py.Group):
                label = name or "/"
                print(f"{indent}+ Group: {label}")
            else:
                print(f"{indent}- Dataset: {name.split('/')[-1]} shape={obj.shape} dtype={obj.dtype}")
            for k, v in obj.attrs.items():
                print(f"{indent}    @ {k}: {v}")
        f.visititems(_print)