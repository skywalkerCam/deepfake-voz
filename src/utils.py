import os
import random
from pathlib import Path

import numpy as np
import torch
import yaml


def _deep_update(base: dict, override: dict) -> dict:
    for k, v in override.items():
        if isinstance(v, dict) and isinstance(base.get(k), dict):
            _deep_update(base[k], v)
        else:
            base[k] = v
    return base


def load_config(path: str) -> dict:
    """Lê config.yaml e, se existir, sobrepõe com config.local.yaml (pessoal, fora do git)."""
    with open(path, encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    local = Path(path).with_name("config.local.yaml")
    if local.exists():
        with open(local, encoding="utf-8") as f:
            _deep_update(cfg, yaml.safe_load(f) or {})
    return cfg


def get_data_root(cfg: dict) -> str:
    """Pasta LA do dataset: variável ASVSPOOF_ROOT > config.local.yaml > config.yaml."""
    root = os.environ.get("ASVSPOOF_ROOT") or cfg["data"]["root"]
    if not Path(root, "ASVspoof2019_LA_cm_protocols").is_dir():
        raise FileNotFoundError(
            f"Dataset não encontrado em '{root}'.\n"
            "Extraia o LA.zip em datasets/ (deve ficar datasets/LA/ASVspoof2019_LA_cm_protocols)\n"
            "ou aponte outro caminho em config.local.yaml (veja config.local.example.yaml).")
    return root


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def get_device(name: str | None = None) -> torch.device:
    if name:
        return torch.device(name)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def count_params(model: torch.nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
