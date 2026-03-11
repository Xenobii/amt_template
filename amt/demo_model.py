import torch
import torch.nn as nn
import torch.nn.functional as func

from amt.model import AMTModel



class DemoModel(AMTModel):
    def __init__(self, **kwargs) -> None:
        super().__init__(**kwargs)

        channels_in = 1

        n_bins    = self.n_bins
        n_pitches = self.max_note - self.min_note

        f = n_bins
        for _ in range(3):
            f = f // 2 - 1

        self.block1 = EncoderBlock(channels_in, 16, 2)
        self.block2 = EncoderBlock(16, 32, 2)
        self.block3 = EncoderBlock(32, 64, 2)
        
        self.convlat = nn.Conv2d(64, 128, kernel_size=(f, 1))

        self.mlp = nn.Sequential(
            nn.Linear(128, n_pitches),
            nn.ReLU()
        )

    def transcription_loss(self, estimate: torch.Tensor, target: torch.Tensor):
        return func.binary_cross_entropy(estimate, target, reduction="mean")

    def forward(self, spec: torch.Tensor) -> torch.Tensor:
        # spec: (B, n_channels, n_bins, n_frames)
        x = self.block1(spec)
        x = self.block2(x)
        x = self.block3(x)                 # (B, 64, f, n_frames)
        x = self.convlat(x)                # (B, 128, 1, n_frames)
        x = x.squeeze(2).permute(0, 2, 1)  # (B, n_frames, 128)
        y = self.mlp(x)                    # (B, n_frames, n_pitches)
        y = y.permute(0, 2, 1)             # (B, n_pitches, n_frames)

        return y

    def forward_train(self, spec: torch.Tensor, roll: torch.Tensor) -> torch.Tensor:
        pred = self(spec)
        return self.transcription_loss(pred, roll)
    
    def inference(self, wav_file: str) -> torch.Tensor:
        spec = self.process_input(wav_file)["spec"] # (B, n_channels, n_bins, n_frames)
        
        with torch.no_grad():
            pred = self(spec) # (B, n_pitches, n_frames)
        
        return pred
    
    def chunked_inference(self, wav_file: str) -> torch.Tensor:
        spec = self.process_input(wav_file)["spec"] # (B, n_channels, n_bins, n_frames)
        
        with torch.no_grad():
            preds = []
            n_chunks = spec.shape[0]
        
            for i in range(n_chunks):
                chunk = spec[i].unsqueeze(0) # (1, n_channels, n_bins, n_frames)
                preds.append(self(chunk).squeeze(0)) # (n_pitches, n_frames)
        
        return torch.stack(preds, dim=0) # (B, n_pitches, n_frames)


# === Modules ===

class EncoderBlock(nn.Module):
    def __init__(
            self,
            in_channels: int,
            out_channels: int,
            stride: int = 2
    ) -> None:
        super().__init__()

        self.block1 = ResidualConv2dBlock(in_channels, in_channels, kernel_size=3, dilation=1)
        self.block2 = ResidualConv2dBlock(in_channels, in_channels, kernel_size=3, dilation=2)
        self.block3 = ResidualConv2dBlock(in_channels, in_channels, kernel_size=3, dilation=3)

        self.hop = stride
        self.win = 2 * stride

        self.sconv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=(self.win, 1), stride=(self.hop, 1)),
            nn.ELU(inplace=True)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Process features
        y = self.block1(x)
        y = self.block2(y)
        y = self.block3(y)

        # Downsample
        y = self.sconv(y)

        return y
    

class ResidualConv2dBlock(nn.Module):
    def __init__(
            self,
            in_channels: int,
            out_channels: int,
            kernel_size: int = 3,
            dilation: int = 1
    ) -> None:
        super().__init__()

        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, padding='same', dilation=dilation),
            nn.ELU(inplace=True)
        )

        self.conv2 = nn.Sequential(
            nn.Conv2d(out_channels, out_channels, kernel_size=1),
            nn.ELU(inplace=True)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Process features
        y = self.conv1(x)
        y = self.conv2(y)

        # Residual
        y = y + x

        return y
   