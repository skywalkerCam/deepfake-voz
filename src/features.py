"""Leitura de áudio e extração de features (LFCC ou log-Mel).

Saída padrão de `FeatureExtractor`: tensor (n_features, n_frames), normalizado
por coeficiente ao longo do tempo (CMVN por locução).
"""
import numpy as np
import soundfile as sf
import torch
import torchaudio.functional as AF
import torchaudio.transforms as T


def load_audio(path: str, sample_rate: int, max_samples: int) -> torch.Tensor:
    wav, sr = sf.read(path, dtype="float32")
    if wav.ndim > 1:
        wav = wav.mean(axis=1)
    wav = torch.from_numpy(wav)
    if sr != sample_rate:
        wav = AF.resample(wav, sr, sample_rate)
    return fix_length(wav, max_samples)


def fix_length(wav: torch.Tensor, max_samples: int) -> torch.Tensor:
    """Corta se for longo; repete o sinal se for curto (padrão nos baselines ASVspoof)."""
    n = wav.shape[0]
    if n >= max_samples:
        return wav[:max_samples]
    reps = int(np.ceil(max_samples / max(n, 1)))
    return wav.repeat(reps)[:max_samples]


class FeatureExtractor:
    def __init__(self, cfg: dict):
        fcfg = cfg["features"]
        self.type = fcfg["type"]
        self.sample_rate = cfg["audio"]["sample_rate"]
        self.max_samples = int(cfg["audio"]["max_seconds"] * self.sample_rate)
        spec = dict(n_fft=fcfg["n_fft"], hop_length=fcfg["hop_length"],
                    win_length=fcfg["win_length"])

        if self.type == "lfcc":
            lc = fcfg["lfcc"]
            self.transform = T.LFCC(sample_rate=self.sample_rate, n_filter=lc["n_filter"],
                                    n_lfcc=lc["n_lfcc"], speckwargs=spec)
            self.deltas = lc.get("deltas", True)
            self.n_features = lc["n_lfcc"] * (3 if self.deltas else 1)
        elif self.type == "logmel":
            self.transform = torch.nn.Sequential(
                T.MelSpectrogram(sample_rate=self.sample_rate, n_mels=fcfg["logmel"]["n_mels"], **spec),
                T.AmplitudeToDB(),
            )
            self.deltas = False
            self.n_features = fcfg["logmel"]["n_mels"]
        else:
            raise ValueError(f"Feature desconhecida: {self.type}")

        self.n_frames = self.max_samples // fcfg["hop_length"] + 1

    @torch.no_grad()
    def __call__(self, wav: torch.Tensor) -> torch.Tensor:
        feat = self.transform(wav)  # (F, T)
        if self.deltas:
            d1 = AF.compute_deltas(feat)
            d2 = AF.compute_deltas(d1)
            feat = torch.cat([feat, d1, d2], dim=0)
        mean = feat.mean(dim=1, keepdim=True)
        std = feat.std(dim=1, keepdim=True) + 1e-5
        return (feat - mean) / std

    def from_file(self, path: str) -> torch.Tensor:
        return self(load_audio(path, self.sample_rate, self.max_samples))

    def cache_tag(self) -> str:
        """Identifica a configuração de features no nome da pasta de cache."""
        return f"{self.type}_{self.n_features}x{self.n_frames}"
