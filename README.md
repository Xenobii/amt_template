# AMT Template: Using `AMTModel` as a Base Class

This repository provides a reusable base class, `AMTModel`, for building downstream automatic music transcription (AMT) models. The base class handles audio/MIDI preprocessing so your subclasses can focus on the neural network architecture and loss functions.

## What `AMTModel` Provides

`amt.model.AMTModel` is a `torch.nn.Module` that encapsulates:

- **Audio I/O & resampling**: loads waveforms with `torchaudio`, converts to mono, and resamples to `target_sr`.
- **Time–frequency transforms**: supports `mel`, `cqt`, `vqt`, `hcqt`, and `hvqt` spectrograms via `nnAudio` / `torchaudio`.
- **Log / dB scaling**: optional power→dB conversion with configurable floor and multiplier.
- **MIDI → piano roll conversion**: builds onset, offset, frame, and velocity rolls from `pretty_midi` using configurable note range and pedal behavior.
- **Time alignment**: pads spectrograms and piano rolls so their time dimensions match.
- **Chunking**: splits long sequences into fixed-length chunks with optional temporal context padding.

You are expected to **subclass** `AMTModel` and implement the model-specific logic:

- `forward(...)`        – core forward pass used for inference.
- `forward_train(...)`  – training-time forward pass / loss computation.
- `inference(...)`      – high-level single-file inference using `process_input`.
- `chunked_inference(...)` – optional long-form inference over chunks.

The file `amt/demo_model.py` contains a complete example subclass (`DemoModel`).

## Subclassing `AMTModel`

### Minimal pattern

A typical model subclass looks like this:

```python
import torch
import torch.nn as nn
import torch.nn.functional as F

from amt.model import AMTModel


class MyAMTModel(AMTModel):
    def __init__(self, **kwargs) -> None:
        # Pass all configuration parameters into the base class
        super().__init__(**kwargs)

        # Example: build a network that consumes spectrograms
        channels_in = 1  # usually 1 log-magnitude channel
        n_bins = self.n_bins
        n_pitches = self.max_note - self.min_note

        self.encoder = nn.Sequential(
            nn.Conv2d(channels_in, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
        )

        self.head = nn.Conv2d(64, n_pitches, kernel_size=1)

    def forward(self, spec: torch.Tensor) -> torch.Tensor:
        """Core forward pass.

        Args:
            spec: (B, n_channels, n_bins, n_frames)

        Returns:
            Tensor (B, n_pitches, n_frames)
        """
        x = self.encoder(spec)
        y = self.head(x)                 # (B, n_pitches, n_bins, n_frames)
        y = y.mean(dim=2)                # collapse frequency axis → (B, n_pitches, n_frames)
        return y

    def forward_train(self, spec: torch.Tensor, roll: torch.Tensor) -> torch.Tensor:
        """Training-time forward + loss.

        Args:
            spec: (B, n_channels, n_bins, n_frames)
            roll: (B, n_pitches, n_frames)
        """
        pred = self(spec)
        return F.binary_cross_entropy(pred, roll, reduction="mean")

    def inference(self, wav_file: str) -> torch.Tensor:
        """Run full-file inference from a waveform path."""
        # Uses AMTModel's preprocessing
        spec = self.process_input(wav_file)["spec"]  # (B, n_channels, n_bins, n_frames)
        with torch.no_grad():
            pred = self(spec)                        # (B, n_pitches, n_frames)
        return pred

    def chunked_inference(self, wav_file: str) -> torch.Tensor:
        """Optional: process long inputs chunk by chunk."""
        spec = self.process_input(wav_file)["spec"]  # (B, n_channels, n_bins, n_frames)
        preds = []
        with torch.no_grad():
            for i in range(spec.shape[0]):
                chunk = spec[i].unsqueeze(0)
                preds.append(self(chunk).squeeze(0))
        return torch.stack(preds, dim=0)  # (B, n_pitches, n_frames)
```

Key points:

- Always call `super().__init__(**kwargs)` so the base preprocessing is configured from your config.
- Expect spectrogram inputs as `(batch, channels, n_bins, n_frames)`.
- Design your network to output piano-roll-like predictions `(batch, n_pitches, n_frames)` that align with the base-class piano rolls.

### Using `process_input` in training

The base class exposes `process_input` for turning paired audio/MIDI into model-ready tensors:

```python
model = MyAMTModel(...)

# From raw files
batch = model.process_input("example.wav", midi_file="example.mid")

spec = batch["spec"]       # (B, n_channels, n_bins, n_frames)
frame = batch["frame"]     # (B, n_pitches, n_frames) if roll_types includes "frame"

loss = model.forward_train(spec, frame)
loss.backward()
```

Downstream pipelines typically wrap this inside a `Dataset` / `DataLoader` rather than calling it directly in the training loop, but the interface is the same.

## Configuration Files for Pipelines

The repository uses YAML configuration files to describe pipelines. A typical config has three top-level sections:

- `model`: which AMT model subclass to instantiate and with what arguments.
- `dataset`: which dataset implementation to use.
- `corpus`: where to find metadata / corpus information.

See `config/demo_config.yaml` for a concrete example.

### Model section

The `model` section describes how to build your subclass. It follows the same argument names as `AMTModel.__init__` and adds a `_target_` field specifying the Python class to instantiate. We recommend using [Hydra](https://github.com/facebookresearch/hydra) to manage these configs and perform the instantiation:

```yaml
model:
  _target_: amt.demo_model.DemoModel  # or your own subclass, e.g. amt.my_model.MyAMTModel

  # AMTModel configuration
  n_bins: 256
  chunk_len: 128
  pad_len: 16
  spec_type: mel        # "mel", "cqt", "vqt", "hcqt", or "hvqt"
  roll_types:
    - frame             # e.g. ["onset", "frame", "velocity"]
  convert_to_log: true
  device: cpu           # or "cuda" if available
  verbose: false

  # Resampling
  default_sr: 44100
  target_sr: 16000

  # Spectrogram parameters
  hop_length: 256
  n_fft: 2048
  win_length: 2048

  # CQT / VQT parameters
  bins_per_octave: 12
  fmin: 18.5

  # dB conversion
  min_value: 1e-8
  multiplier: 10.0

  # MIDI / piano-roll parameters
  min_note: A0
  max_note: C8
  precise_onsets: true
  precise_offsets: true
  include_pedal: true
```

With Hydra, you can keep `_target_` in the config and let Hydra construct the object automatically.

### Dataset section

Datasets are also configured via `_target_` to keep pipelines generic. The demo config uses `MaestroDataset`:

```yaml
dataset:
  _target_: amt.datasets.MaestroDataset
  root: dataset  # path to the dataset root
```

With Hydra, the dataset can also be instantiated directly from its `_target_` entry.

### Corpus section

The `corpus` section describes where to find additional metadata (e.g., cached paths, composer names) used by the dataset or training loop:

```yaml
corpus:
  corpus_file: corpus/corpus.h5
  attributes:
    - midi_filename
    - audio_filename
    - canonical_composer
```

This is optional and specific to your dataset implementation; the `AMTModel` base class itself does not depend on it.

## End-to-End Example

Putting it all together, a minimal training setup with a custom subclass might look like:

1. **Define your model** in `amt/my_model.py` as a subclass of `AMTModel` (see `MyAMTModel` above).
2. **Create a config** (e.g., `config/my_experiment.yaml`):

   ```yaml
   model:
     _target_: amt.my_model.MyAMTModel
     n_bins: 256
     chunk_len: 128
     pad_len: 16
     spec_type: mel
     roll_types: [frame]
     convert_to_log: true
     device: cuda
     verbose: false

     default_sr: 44100
     target_sr: 16000
     hop_length: 256

     min_note: A0
     max_note: C8
     precise_onsets: true
     precise_offsets: true
     include_pedal: true

   dataset:
     _target_: amt.datasets.MaestroDataset
     root: /path/to/maestro

   corpus:
     corpus_file: corpus/corpus.h5
     attributes: [midi_filename, audio_filename]
   ```

3. **Training script sketch with Hydra** (simplified):

   ```python
   import hydra
   from hydra.utils import instantiate
   from omegaconf import DictConfig

   @hydra.main(config_path="config", config_name="my_experiment", version_base=None)
   def main(cfg: DictConfig) -> None:
     # Instantiate model and dataset directly from config
     model = instantiate(cfg.model)
     dataset = instantiate(cfg.dataset)

     # Example: access optional corpus config
     corpus_cfg = cfg.get("corpus")

     # Your DataLoader + training loop here, using
     # model.forward_train(spec, roll) inside the loop.

   if __name__ == "__main__":
     main()
   ```

This layout lets you reuse `AMTModel` for many different architectures and datasets while keeping your preprocessing and configs consistent across AMT pipelines, and Hydra keeps configuration management scalable as experiments grow.
