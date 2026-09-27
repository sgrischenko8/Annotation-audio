from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
import librosa
import numpy as np


@dataclass
class AudioClip:
    y: np.ndarray
    sr: int
    path: Path

    @property
    def duration(self) -> float:
        return float(librosa.get_duration(y=self.y, sr=self.sr))


def load_audio(path: str | Path, sr: int | None = None, mono: bool = True) -> AudioClip:
    path = Path(path)
    y, loaded_sr = librosa.load(path, sr=sr, mono=mono)
    return AudioClip(y=y, sr=loaded_sr, path=path)


def compute_spectrogram(
    clip: AudioClip,
    n_fft: int = 2048,
    hop_length: int = 512,
    kind: str = "mel",
    n_mels: int = 128,
    fmax: float | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    if kind == "mel":
        S = librosa.feature.melspectrogram(
            y=clip.y, sr=clip.sr, n_fft=n_fft, hop_length=hop_length,
            n_mels=n_mels, fmax=fmax,
        )
        S_db = librosa.power_to_db(S, ref=np.max)
        freqs = librosa.mel_frequencies(n_mels=n_mels, fmax=fmax or clip.sr / 2)
    elif kind == "stft":
        S = np.abs(librosa.stft(clip.y, n_fft=n_fft, hop_length=hop_length))
        S_db = librosa.amplitude_to_db(S, ref=np.max)
        freqs = librosa.fft_frequencies(sr=clip.sr, n_fft=n_fft)
    else:
        raise ValueError(f"Unknown type of spectrogram: {kind!r} (expected 'mel' or 'stft')")

    times = librosa.frames_to_time(np.arange(S_db.shape[1]), sr=clip.sr, hop_length=hop_length)
    return S_db, times, freqs