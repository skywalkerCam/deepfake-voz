"""Gera o subconjunto aleatório (uma vez só) e salva em CSV.

Uso:
    python -m src.make_subset --config config.yaml

Rode UMA vez e versione os CSVs em data/subset/: assim todos os modelos da
equipe são treinados e avaliados exatamente nos mesmos áudios.
"""
import argparse
from pathlib import Path

from .data import read_protocol, sample_subset
from .utils import get_data_root, load_config


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="config.yaml")
    ap.add_argument("--force", action="store_true", help="sobrescreve CSVs existentes")
    args = ap.parse_args()

    cfg = load_config(args.config)
    out_dir = Path(cfg["data"]["subset_dir"])
    out_dir.mkdir(parents=True, exist_ok=True)

    root = get_data_root(cfg)
    print(f"Dataset: {root}")
    for split, sizes in cfg["data"]["subset"].items():
        out = out_dir / f"{split}.csv"
        if out.exists() and not args.force:
            print(f"[{split}] {out} já existe — use --force para refazer")
            continue
        df = read_protocol(root, split)
        sub = sample_subset(df, sizes["bonafide"], sizes["spoof"], seed=cfg["seed"])
        missing = [p for p in sub.path if not (Path(root) / p).exists()]
        if missing:
            print(f"[{split}] AVISO: {len(missing)} arquivos não encontrados (ex.: {missing[0]})")
        sub.to_csv(out, index=False)
        counts = sub.key.value_counts().to_dict()
        print(f"[{split}] {len(sub)} amostras {counts} | ataques: {sub[sub.label == 1].system.nunique()} -> {out}")


if __name__ == "__main__":
    main()
