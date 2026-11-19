"""Junta os metrics.json de todos os modelos em uma tabela comparativa.

Uso:
    python -m src.compare            # imprime e salva results/comparison.csv e .md
"""
import argparse
import json
from pathlib import Path

import pandas as pd


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", default="results")
    ap.add_argument("--split", default="eval", choices=["dev", "eval"])
    args = ap.parse_args()

    rows = []
    for f in sorted(Path(args.results).glob("*/metrics.json")):
        r = json.loads(f.read_text(encoding="utf-8"))
        m = r[args.split]
        rows.append({
            "modelo": r["model"],
            "precision": m["precision"],
            "recall": m["recall"],
            "f1": m["f1"],
            "eer_%": m["eer"] * 100,
            "accuracy": m["accuracy"],
            "roc_auc": m["roc_auc"],
            "params": r["n_params"],
            "épocas": r["epochs_run"],
            "tempo_treino_s": r["train_time_s"],
        })
    if not rows:
        print("Nenhum results/*/metrics.json encontrado.")
        return
    df = pd.DataFrame(rows).sort_values("eer_%")
    out = Path(args.results)
    df.to_csv(out / f"comparison_{args.split}.csv", index=False)
    md = df.to_markdown(index=False, floatfmt=".4f")
    (out / f"comparison_{args.split}.md").write_text(md, encoding="utf-8")
    print(md)


if __name__ == "__main__":
    main()
