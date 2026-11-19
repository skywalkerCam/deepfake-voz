"""Protocolos do ASVspoof 2019 LA, amostragem do subconjunto e Dataset PyTorch.

Convenção de rótulos (usada em todo o projeto):
    spoof (deepfake) = 1  -> classe POSITIVA
    bonafide         = 0
"""
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

from .features import FeatureExtractor

SPLITS = {
    "train": ("ASVspoof2019_LA_train", "ASVspoof2019.LA.cm.train.trn.txt"),
    "dev": ("ASVspoof2019_LA_dev", "ASVspoof2019.LA.cm.dev.trl.txt"),
    "eval": ("ASVspoof2019_LA_eval", "ASVspoof2019.LA.cm.eval.trl.txt"),
}
LABELS = {"bonafide": 0, "spoof": 1}


def read_protocol(root: str, split: str) -> pd.DataFrame:
    """Lê o protocolo CM: SPEAKER  ARQUIVO  -  SISTEMA  CHAVE."""
    folder, proto = SPLITS[split]
    path = Path(root) / "ASVspoof2019_LA_cm_protocols" / proto
    df = pd.read_csv(path, sep=" ", header=None,
                     names=["speaker", "utt_id", "_", "system", "key"])
    df = df.drop(columns="_")
    df["label"] = df["key"].map(LABELS)
    # caminho RELATIVO à pasta LA: os CSVs do subset funcionam em qualquer computador
    df["path"] = [f"{folder}/flac/{u}.flac" for u in df["utt_id"]]
    df["split"] = split
    return df


def sample_subset(df: pd.DataFrame, n_bonafide: int, n_spoof: int, seed: int) -> pd.DataFrame:
    """Amostra aleatória: bonafide simples; spoof estratificado por sistema de ataque."""
    rng = np.random.RandomState(seed)
    bona = df[df.label == 0]
    spoof = df[df.label == 1]
    bona = bona.sample(n=min(n_bonafide, len(bona)), random_state=rng)

    n_spoof = min(n_spoof, len(spoof))
    systems = sorted(spoof.system.unique())
    # distribui n_spoof igualmente entre os ataques (o resto vai para os primeiros)
    base, extra = divmod(n_spoof, len(systems))
    parts = []
    for i, sysid in enumerate(systems):
        g = spoof[spoof.system == sysid]
        k = min(base + (1 if i < extra else 0), len(g))
        parts.append(g.sample(n=k, random_state=rng))
    out = pd.concat([bona, *parts]).sample(frac=1.0, random_state=rng)
    return out.reset_index(drop=True)


class ASVspoofDataset(Dataset):
    """Retorna (features (F, T) float32, label float32, utt_id).

    As features são calculadas uma vez e salvas em cache (.npy).
    """

    def __init__(self, df: pd.DataFrame, extractor: FeatureExtractor, cache_dir: str, root: str):
        self.df = df.reset_index(drop=True)
        self.root = Path(root)
        self.extractor = extractor
        self.cache_dir = Path(cache_dir) / extractor.cache_tag()
        self.cache_dir.mkdir(parents=True, exist_ok=True)

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int):
        row = self.df.iloc[idx]
        cache = self.cache_dir / f"{row.utt_id}.npy"
        if cache.exists():
            feat = torch.from_numpy(np.load(cache))
        else:
            feat = self.extractor.from_file(str(self.root / row.path))
            np.save(cache, feat.numpy().astype(np.float32))
        return feat.float(), torch.tensor(row.label, dtype=torch.float32), row.utt_id


def load_subset(subset_dir: str, split: str) -> pd.DataFrame:
    path = Path(subset_dir) / f"{split}.csv"
    if not path.exists():
        raise FileNotFoundError(
            f"{path} não existe. Rode antes: python -m src.make_subset --config config.yaml")
    return pd.read_csv(path)
