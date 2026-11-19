"""Verifica se um modelo respeita o contrato, sem precisar do dataset.

Uso:
    python -m scripts.check_model --model rnn
"""
import argparse

import torch

from src.features import FeatureExtractor
from src.models import build_model
from src.utils import count_params, load_config


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--config", default="config.yaml")
    args = ap.parse_args()

    cfg = load_config(args.config)
    ext = FeatureExtractor(cfg)
    model = build_model(args.model, ext.n_features, ext.n_frames, cfg)

    x = torch.randn(4, ext.n_features, ext.n_frames)
    y = torch.tensor([0.0, 1.0, 0.0, 1.0])
    out = model(x)
    assert out.shape == (4,), f"saída deve ter shape (batch,), veio {tuple(out.shape)}"

    loss = torch.nn.BCEWithLogitsLoss()(out, y)
    loss.backward()
    no_grad = [n for n, p in model.named_parameters() if p.requires_grad and p.grad is None]
    assert not no_grad, f"parâmetros sem gradiente: {no_grad[:5]}"

    model.eval()
    with torch.no_grad():
        out_eval = model(torch.randn(1, ext.n_features, ext.n_frames))
    assert out_eval.shape == (1,), "batch de tamanho 1 deve retornar shape (1,)"

    print(f"OK: '{args.model}' | entrada {tuple(x.shape)} -> saída {tuple(out.shape)} | "
          f"params: {count_params(model):,}")


if __name__ == "__main__":
    main()
