"""Registro de modelos.

CONTRATO que todo modelo deve seguir:
    __init__(self, n_features: int, n_frames: int, **hparams)
        hparams vêm de config.yaml -> models.<nome>
    forward(self, x) -> logits
        x: tensor (batch, n_features, n_frames)
        retorno: tensor (batch,) com LOGITS (sem sigmoid); positivo = spoof

Verifique seu modelo com:  python -m scripts.check_model --model <nome>
"""
from .cnn import CNNClassifier
from .rnn import RNNClassifier
from .transformer import TransformerClassifier

MODELS = {
    "cnn": CNNClassifier,                  # implementado
    "rnn": RNNClassifier,                  # TODO: integrante 2
    "transformer": TransformerClassifier,  # TODO: integrante 3
}


def build_model(name: str, n_features: int, n_frames: int, cfg: dict):
    if name not in MODELS:
        raise ValueError(f"Modelo '{name}' não registrado. Opções: {list(MODELS)}")
    hparams = cfg.get("models", {}).get(name, {}) or {}
    return MODELS[name](n_features=n_features, n_frames=n_frames, **hparams)
