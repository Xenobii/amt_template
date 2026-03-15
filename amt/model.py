import torch
import torch.nn as nn
import torchaudio
import torch.nn.functional as func
import numpy as np
from nnAudio.features import CQT2010v2
from nnAudio.features import VQT
import pretty_midi
from typing import Tuple, Optional, List, Dict



class AMTModel(nn.Module):
    def __init__(
            self,

            # model dimensions and defaults
            n_bins: int = 256,
            chunk_len: int = 128,
            pad_len: int = 32,
            spec_type: str = "cqt",
            roll_types: List = ["frame"],
            convert_to_log: bool = True,
            device: str = "cpu",
            verbose: bool = True,
            
            # resampling rate
            default_sr: int = 44100,
            target_sr: int = 16000,
            
            # spectrogram parameters
            hop_length: int = 256,

            # cqt/vqt specifics
            bins_per_octave: int = 36,
            fmin: float = 27.5,

            # hcqt specifics
            harmonics: tuple = [0.5, 1, 2, 3, 4, 5],
            
            # mel specifics
            n_fft: int = 2048,
            win_length: int = 2048,
            
            # dB conversion
            min_value: float = 1e-8,
            multiplier: float = 10.0,

            # midi retrieval
            min_note: str = "A0",
            max_note: str = "C8",
            precise_onsets: bool = True,
            precise_offsets: bool = True,
            include_pedal: bool = True,

            **kwargs
    ) -> None:
        super().__init__()

        self.verbose = verbose

        if torch.cuda.is_available():
            self.device = torch.device(device)
        else:
            if self.verbose: 
                print("Cuda is not available")
            self.device = torch.device("cpu")
        if self.verbose:
            print(f"Model loaded on {self.device}")
        
        # --- model general settings and dimensions ---
        self.n_bins           = n_bins
        self.chunk_len        = chunk_len
        self.pad_len          = pad_len
        self.spec_type        = spec_type
        self.convert_to_log   = convert_to_log
        self.roll_types       = roll_types

        if verbose:
            print("--- Model settings ---")
            print(f"{'n_bins':15s}: {self.n_bins}")
            print(f"{'chunk_len':15s}: {self.chunk_len}")
            print(f"{'pad_len':15s}: {self.pad_len}")
            print(f"{'spec_type':15s}: {self.spec_type}")
            print(f"{'convert_to_log':15s}: {self.convert_to_log}")
            print(f"{'roll_types':15s}: {self.roll_types}")

        # --- resampling ---
        self.target_sr  = target_sr
        self.default_sr = default_sr
        self.resampler  = torchaudio.transforms.Resample(self.default_sr, self.target_sr).to(self.device)
        
        if verbose:
            print(f"{'target_sr':15s}: {self.target_sr}")
            print(f"{'default_sr':15s}: {self.default_sr}")
            print(f"{'hop_length':15s}: {hop_length}")

        # --- mel ---
        if self.spec_type == "mel":
            self.mel = torchaudio.transforms.MelSpectrogram(
                sample_rate=target_sr,
                n_fft=n_fft,
                win_length=win_length,
                hop_length=hop_length,
                pad_mode="constant",
                n_mels=n_bins,
                norm="slaney"
            ).to(self.device)
        
            if verbose:
                print(f"{'n_fft':15s}: {n_fft}")
                print(f"{'win_length':15s}: {win_length}")

        # --- cqt ---
        if self.spec_type == "cqt":
            self.cqt = CQT2010v2(
                sr=target_sr,
                hop_length=hop_length,
                fmin=fmin,
                n_bins=n_bins,
                bins_per_octave=bins_per_octave,
                pad_mode="constant",
                verbose=False
            ).to(self.device)

            if verbose:
                print(f"{'bins_per_octave':15s}: {bins_per_octave}")
                print(f"{'fmin':15s}: {fmin}")
        
        # --- vqt ---
        if self.spec_type == "vqt":
            self.vqt = VQT(
                sr=target_sr,
                hop_length=hop_length,
                fmin=fmin,
                n_bins=n_bins,
                bins_per_octave=bins_per_octave,
                pad_mode="constant",
                verbose=False
            ).to(self.device)

            if verbose:
                print(f"{'bins_per_octave':15s}: {bins_per_octave}")
                print(f"{'fmin':15s}: {fmin}")

        # --- hcqt ---
        if self.spec_type == "hcqt":
            self.hcqt = nn.ModuleList(
                [
                    CQT2010v2(
                        sr=target_sr,
                        hop_length=hop_length,
                        fmin=fmin * h,
                        n_bins=n_bins,
                        bins_per_octave=bins_per_octave,
                        pad_mode="constant",
                        verbose=False
                    )
                    for h in harmonics
                ]
            ).to(self.device)

            if verbose:
                print(f"{'bins_per_octave':15s}: {bins_per_octave}")
                print(f"{'fmin':15s}: {fmin}")
                print(f"{'harmonics':15s}: {harmonics}")
        
        # --- hvqt ---
        if self.spec_type == "hvqt":
            self.hvqt = nn.ModuleList(
                [
                    VQT(
                        sr=target_sr,
                        hop_length=hop_length,
                        fmin=fmin * h,
                        n_bins=n_bins,
                        bins_per_octave=bins_per_octave,
                        pad_mode="constant",
                        verbose=False
                    )
                    for h in harmonics
                ]
            ).to(self.device)

            if verbose:
                print(f"{'bins_per_octave':15s}: {bins_per_octave}")
                print(f"{'fmin':15s}: {fmin}")
                print(f"{'harmonics':15s}: {harmonics}")

        # --- dB conversion ---
        self.multiplier = multiplier
        self.min_value  = min_value
        if self.convert_to_log:
            self.pad_value = self.multiplier * np.log10(self.min_value)
        else:
            self.pad_value = self.min_value
        
        if verbose:
            print(f"{'min_value':15s}: {min_value}")
            print(f"{'multiplier':15s}: {multiplier}")
        
        # --- midi retrieval ---
        self.fs = int(target_sr / hop_length)
        self.min_note = pretty_midi.note_name_to_number(min_note)
        self.max_note = pretty_midi.note_name_to_number(max_note)
        self.precise_onsets = precise_onsets
        self.precise_offsets = precise_offsets
        self.onset_window = (0.5, 1.0, 0.7, 0.3, 0.1)
        self.onset_kernel = torch.tensor(self.onset_window).to(self.device)
        self.offset_window = (0.1, 0.25, 0.5, 0.85, 1.0, 0.67, 0.33)
        self.offset_kernel = torch.tensor(self.offset_window).to(self.device)
        if include_pedal:
            self.pedal_threshold = 64
        else:
            self.pedal_threshold = 128
        
        if verbose:
            print(f"{'min_note':15s}: {min_note}={self.min_note}")
            print(f"{'max_note':15s}: {max_note}={self.max_note}")
            print(f"{'precise_onsets':15s}: {precise_onsets}")
            print(f"{'precise_offsets':15s}: {precise_offsets}")
            print(f"{'include_pedal':15s}: {include_pedal}")

    def get_spectrogram(self, wav_file: str, spec_type: str = "cqt", convert_to_log: bool = True) -> torch.Tensor:
        """
        Load an audio file, compute the requested spectrogram, and optionally convert to dB.

        Args:
            wav_file: Path to the waveform file.
            spec_type: One of 'mel','cqt','vqt','hcqt','hvqt'.
            convert_to_log: If True, convert power to dB.

        Returns:
            Spectrogram tensor on model device.
            Shapes:
              - `mel/cqt/vqt`: (channels, n_bins, time)
              - `hcqt/hvqt`: (n_harmonics, n_bins, time)
        """
        # load audio
        wave, sr = torchaudio.load(wav_file)
        wave = wave.to(self.device)

        # to mono
        if len(wave.shape) != 1:
            wave = wave.mean(dim=0, keepdim=True)

        # resample
        if sr != self.target_sr:
            if sr != self.default_sr:
                resampler = torchaudio.transforms.Resample(sr, self.target_sr).to(wave.device)
            else:
                resampler = self.resampler
            wave = resampler(wave)

        # create spec
        if spec_type == "mel":
            spec = self._get_mel(wave)
        elif spec_type == "cqt":
            spec = self._get_cqt(wave)
        elif spec_type == "vqt":
            spec = self._get_vqt(wave)
        elif spec_type == "hcqt":
            spec = self._get_hcqt(wave)
        elif spec_type == "hvqt":
            spec = self._get_hvqt(wave)
        else:
            raise ValueError(f"Invalid spec_type: {spec_type}")
        
        # to log
        if convert_to_log:
            spec = self._power_to_db(spec)

        return spec
    
    def _power_to_db(self, x: torch.Tensor) -> torch.Tensor:
        """Convert power spectrogram to dB scale.
            Args: 
                x: Power spectrogram tensor.
            Returns: 
                dB-scaled tensor of same shape as `x`.
        """
        return self.multiplier * torch.log10(x.clamp(min=self.min_value))
    
    def _db_to_power(self, x: torch.Tensor) -> torch.Tensor:
        """Convert dB spectrogram back to power.
        Args:
            x: dB spectrogram tensor.
        Returns:
            Power spectrogram tensor of same shape as `x`.
        """
        return torch.pow(10.0, x / self.multiplier)
    
    def _get_mel(self, x: torch.Tensor) -> torch.Tensor:
        """Compute Mel spectrogram.
        Args:
            x: Waveform tensor (channels, samples).
        Returns:
            Mel spectrogram: (channels, n_mels, time)
        """
        return self.mel(x)
    
    def _get_cqt(self, x: torch.Tensor) -> torch.Tensor:
        """Compute Constant-Q transform (CQT).
        Args:
            x: Waveform tensor (channels, samples).
        Returns:
            CQT spectrogram: (channels, n_bins, time)
        """
        return self.cqt(x)
    
    def _get_vqt(self, x: torch.Tensor) -> torch.Tensor:
        """Compute Variable-Q transform (VQT).
        Args:
            x: Waveform tensor (channels, samples).
        Returns:
            VQT spectrogram: (channels, n_bins, time)
        """
        return self.vqt(x)
    
    def _get_hcqt(self, x: torch.Tensor) -> torch.Tensor:
        """Compute harmonic CQT stack.
        Args:
            x: Waveform tensor (channels, samples).
        Returns:
            Tensor of stacked harmonic CQTs: (n_harmonics, n_bins, time)
        """
        specs = []
        for cqt in self.hcqt:
            spec = cqt(x).squeeze(0)
            specs.append(spec)
        return torch.stack(specs, dim=0)
    
    def _get_hvqt(self, x: torch.Tensor) -> torch.Tensor:
        """Compute harmonic VQT stack.
        Args:
            x: Waveform tensor (channels, samples).
        Returns:
            Tensor of stacked harmonic VQTs: (n_harmonics, n_bins, time)
        """
        specs = []
        for vqt in self.hvqt:
            spec = vqt(x).squeeze(0)
            specs.append(spec)
        return torch.stack(specs, dim=0)
    
    # === Midi retrieval ===

    def get_piano_roll(self, midi_file: str, roll_types: Optional[List] = None) -> torch.Tensor:
        """Load a MIDI file and return requested piano roll channels.
        Args:
            midi_file: Path to MIDI file.
            roll_types: List of roll types to include (e.g. ['onset','frame','velocity']).
        Returns:
            Stacked roll tensor: (channels, n_pitches, time) where `channels` matches len(roll_types).
        """
        # load midi
        midi = pretty_midi.PrettyMIDI(midi_file)

        rolls = []
        if "onset" in roll_types:
            rolls.append(self._get_onset_roll(midi, include_velocity=False))
        if "onset_with_velocity" in roll_types:
            rolls.append(self._get_onset_roll(midi, include_velocity=True))
        if "offset" in roll_types:
            rolls.append(self._get_offset_roll(midi, include_velocity=False)) 
        if "offset_with_velocity" in roll_types:
            rolls.append(self._get_offset_roll(midi, include_velocity=True)) 
        if "frame" in roll_types:    
            rolls.append(self._get_frame_roll(midi)) 
        if "velocity" in roll_types:
            rolls.append(self._get_velocity_roll(midi))

        # match time
        n_frames = max(r.shape[-1] for r in rolls)
        for i, r in enumerate(rolls):
            pad_len = n_frames - r.shape[-1]

            if pad_len > 0:
                rolls[i] = func.pad(r, (0, pad_len), value=0.0)
        
        # stack for parallel processing
        roll = torch.stack(rolls, dim=0)

        return roll # (channels, n_pitches, time)
    
    def _get_pedal_intervals(self, instrument):
        """Extract sustain pedal on/off time intervals for an instrument.
        Args:
            instrument: pretty_midi.Instrument instance.
        Returns:
            List of (start, end) times for pedal-on intervals.
        """
        intervals = []
        pedal_on = False
        start = None

        for cc in instrument.control_changes:
            if cc.number != 64:
                continue

            if cc.value >= self.pedal_threshold and not pedal_on:
                pedal_on = True
                start = cc.time

            elif cc.value < self.pedal_threshold and pedal_on:
                pedal_on = False
                intervals.append((start, cc.time))

        if pedal_on:
            intervals.append((start, float("inf")))

        return intervals

    def _get_onset_roll(self, midi: pretty_midi.PrettyMIDI, include_velocity: bool = True) -> torch.Tensor:
        """Create an onset piano roll from MIDI.
        Args:
            midi: pretty_midi.PrettyMIDI instance.
            include_velocity: If True, scale onsets by note velocity.
        Returns:
            Onset roll tensor: (n_pitches, time)
        """
        if self.precise_onsets:
            end_time = midi.get_end_time()
            n_frames = int(np.ceil(end_time * self.fs)) + len(self.onset_window)
            n_pitches = self.max_note - self.min_note + 1

            # create matrix
            onset_roll = torch.zeros((n_pitches, n_frames), dtype=torch.float32, device=self.device)

            # iterate over note onsets
            for instrument in midi.instruments:
                for note in instrument.notes:
                    pitch = note.pitch
                    if pitch < self.min_note or pitch > self.max_note:
                        continue
                        
                    pitch_idx = pitch - self.min_note
                    onset_frame = int(note.start * self.fs)

                    scale = note.velocity / 127.0 if include_velocity else 1.0

                    for k, val in enumerate(self.onset_kernel):
                        frame = onset_frame + k
                        if frame < n_frames:
                            onset_roll[pitch_idx, frame] = max(onset_roll[pitch_idx, frame], val*scale)

        else:
            # get piano roll
            if include_velocity:
                piano_roll = self._get_velocity_roll(midi) / 127.0
            else:
                piano_roll = self._get_frame_roll(midi)

            # create onset roll
            onset_roll = ((piano_roll[:, 1:] - piano_roll[:, :-1])>0).float()
            onset_roll = func.pad(onset_roll, (1,0))

        return onset_roll # (n_pitches, time)

    def _get_offset_roll(self, midi: pretty_midi.PrettyMIDI, include_velocity: bool = True) -> torch.Tensor:
        """Create an offset piano roll from MIDI.
        Args:
            midi: pretty_midi.PrettyMIDI instance.
            include_velocity: If True, scale offsets by note velocity.
        Returns:
            Offset roll tensor: (n_pitches, time)
        """
        if self.precise_offsets:
            end_time = midi.get_end_time()
            n_frames = int(np.ceil(end_time * self.fs)) + len(self.offset_window)
            n_pitches = self.max_note - self.min_note + 1

            # create matrix
            offset_roll = torch.zeros((n_pitches, n_frames), dtype=torch.float32, device=self.device)

            # iterate over note offsets
            for instrument in midi.instruments:
                if instrument.is_drum:
                    continue

                pedal_intervals = self._get_pedal_intervals(instrument)

                for note in instrument.notes:
                    pitch = note.pitch
                    if pitch < self.min_note or pitch > self.max_note:
                        continue
                        
                    pitch_idx = pitch - self.min_note
                    offset_time = note.end
                    for p_start, p_end in pedal_intervals:
                        if p_start <= note.end < p_end:
                            offset_time = p_end
                            break

                    offset_frame = int(offset_time * self.fs)

                    scale = note.velocity / 127.0 if include_velocity else 1.0

                    for k, val in enumerate(self.offset_kernel):
                        frame = offset_frame + k
                        if frame < n_frames:
                            offset_roll[pitch_idx, frame] = max(offset_roll[pitch_idx, frame], val*scale)

        else:
            # get piano roll
            if include_velocity:
                piano_roll = self._get_velocity_roll(midi) / 127.0
            else:
                piano_roll = self._get_frame_roll(midi)

            # create onset roll
            offset_roll = ((piano_roll[:, :-1] - piano_roll[:, 1:]) > 0).float()
            offset_roll = func.pad(offset_roll, (0, 1), value=0.0)
        
        return offset_roll # (n_pitches, time)
    
    def _get_frame_roll(self, midi: pretty_midi.PrettyMIDI) -> torch.Tensor:
        """Return binary frame-level piano roll from MIDI.
        Args:
            midi: pretty_midi.PrettyMIDI instance.
        Returns:
            Binary frame roll tensor: (n_pitches, time)
        """
        # get piano roll
        frame_roll = midi.get_piano_roll(fs=self.fs, pedal_threshold=self.pedal_threshold)
        frame_roll = torch.from_numpy(frame_roll>0).float().to(self.device)
        
        # crop
        frame_roll = frame_roll[self.min_note:self.max_note+1, :]

        return frame_roll # (n_pitches, time)
    
    def _get_velocity_roll(self, midi: pretty_midi.PrettyMIDI) -> torch.Tensor:
        """Return velocity-scaled piano roll from MIDI.
        Args:
            midi: pretty_midi.PrettyMIDI instance.
        Returns:
            Velocity roll tensor: (n_pitches, time) with values 0-127.
        """
        # convert to piano roll
        velocity_roll = midi.get_piano_roll(fs=self.fs, pedal_threshold=self.pedal_threshold)

        # to torch
        velocity_roll = torch.from_numpy(velocity_roll).float().to(self.device)
        
        # crop
        velocity_roll = velocity_roll[self.min_note:self.max_note+1, :]

        return velocity_roll # (n_pitches, time)

    # === Chunking ===

    def _match_input_times(self, spec: torch.Tensor, roll: torch.Tensor, pad_value: float = 0.0) -> Tuple[torch.Tensor, torch.Tensor]:
        """Pad spectrogram or piano roll so their time dimensions match.
        Args:
            spec: Spectrogram tensor (channels, n_bins, time).
            roll: Piano roll tensor (channels, n_pitches, time).
            pad_value: Value used to pad spectrogram when it's shorter.
        Returns:
            Tuple (spec, roll) with matched time dimension.
        """
        # spec: (channels, n_bins, time)
        # roll: (channels, n_pitches, time)
            
        # handle both spectrogram and piano roll
        time_spec = spec.shape[2]
        time_roll = roll.shape[2]
        
        # match times
        if time_spec > time_roll:
            roll = func.pad(roll, (0, time_spec-time_roll), value=0.0)
        elif time_roll > time_spec:
            if self.verbose:
                print(f"Piano roll time {time_roll} > spectrogram time {time_spec}")
            spec = func.pad(spec, (0, time_roll-time_spec), value=pad_value)

        return spec, roll

    def _chunk_input(self, x: torch.Tensor, chunk_len: int, pad_len: int, pad_value: float) -> torch.Tensor:
        """Split a time-series tensor into fixed-length chunks with optional padding.
        Args:
            x: Input tensor (channels, n_bins, n_frames).
            chunk_len: Frames per chunk (without pad_len).
            pad_len: Frames of padding on each side of a chunk.
            pad_value: Value used for padding.
        Returns:
            Chunked tensor: (n_chunks, channels, n_bins, chunk_len + 2*pad_len)
        """
        # x: (channels, n_bins, n_frames)
        channels, n_bins, n_frames = x.shape
        
        # pad end to divide by chunks
        pad_end = (chunk_len - (n_frames % chunk_len)) % chunk_len
        x = func.pad(x, (0, pad_end), value=pad_value)

        # padding
        if pad_len > 0:
            x = func.pad(x, (pad_len, pad_len), value=pad_value)

        # chunk
        if pad_len > 0:
            x = x.unfold(dimension=2, size=chunk_len+2*pad_len, step=chunk_len)
        else:
            n_chunks = x.shape[-1] // chunk_len
            x = x.reshape(channels, n_bins, n_chunks, chunk_len)

        return x.permute(2, 0, 1, 3) # (n_chunks, channels, n_bins, chunk_len + 2*pad_len)

    # === Preprocessing ===
    
    def process_input(self, wav_file: str, midi_file: Optional[str] = None) -> Dict:
        """Process audio (and optional MIDI) into chunked spectrograms and piano rolls.
        Args:
            wav_file: Path to waveform file.
            midi_file: Optional path to MIDI file. If provided, corresponding rolls are returned.
        Returns:
            Dictionary with key 'spec' and additional keys for each roll type when `midi_file` is given.
            - `spec`: (n_chunks, channels, n_bins, chunk_len + 2*pad_len)
            - roll entries: (n_chunks, n_pitches, chunk_len + 2*pad_len) or similar per roll type
        """
        with torch.no_grad():
            output = {}
            if midi_file is not None:
                spec = self.get_spectrogram(wav_file, self.spec_type, self.convert_to_log)  # (channels, n_bins, n_frames)
                roll = self.get_piano_roll(midi_file, self.roll_types)                      # (channels, n_pitches, n_frames)

                # match input times
                spec, roll = self._match_input_times(spec, roll, self.pad_value)
                
                # create chunks
                spec = self._chunk_input(spec, self.chunk_len, self.pad_len, pad_value=self.pad_value)  # (n_chunks, channels, n_bins, n_frames_per_chunk)
                roll = self._chunk_input(roll, self.chunk_len, self.pad_len, pad_value=0.0)             # (n_chunks, channels, n_bins, n_frames_per_chunk)
                
                # create object
                output["spec"] = spec
                for i in range(len(self.roll_types)):
                    roll_type = self.roll_types[i]
                    output[roll_type] = roll[:, i, :, :]

            else:
                spec = self.get_spectrogram(wav_file, self.spec_type, self.convert_to_log)    # (channels, n_bins, n_frames)
                spec = self._chunk_input(spec, self.chunk_len, self.pad_len, self.pad_value)  # (n_chunks, n_channels, n_bins, n_frames_per_chunk)
                output["spec"] = spec

        return output
    
    # === Forward methods ===

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Model forward for inference. To be implemented by subclasses.
        Args:
            x: Input tensor (shape depends on model architecture).
        Returns:
            Model output tensor.
        """
        return NotImplementedError
    
    def forward_train(self, x: torch.Tensor) -> torch.Tensor:
        """Model forward used during training. To be implemented by subclasses.
        Args:
            x: Input tensor (shape depends on model architecture).
        Returns:
            Training-time outputs (e.g., logits, losses, metrics).
        """
        return NotImplementedError

    # === Inference methods ===
    
    def inference(self, wav_file: str):
        """High-level inference API: run full-file inference and return predictions.
        Args:
            wav_file: Path to waveform file.
        Returns:
            Model-dependent inference result.
        """
        return NotImplementedError
    
    def chunked_inference(self, wav_file: str):
        """Run inference on chunked input to support long audio processing.
        Args:
            wav_file: Path to waveform file.
        Returns:
            Model-dependent chunked predictions.
        """
        return NotImplementedError
        
    # === Utilities ===

    def print_num_params(self):
        total = sum(p.numel() for p in self.parameters())
        print(f"{'Total number of parameters':30s}: {total:,}")
        """Print the total number of trainable parameters in the model."""