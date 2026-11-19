"""Treino + avaliação de um modelo registrado.

Uso:
    python -m src.train --model cnn
    python -m src.train --model rnn --epochs 20 --device cuda

Protocolo:
    1. treina no subset de train;
    2. early stopping pelo EER no dev (salva o melhor checkpoint);
    3. limiar de decisão = limiar do EER no dev;
    4. aplica esse limiar no eval -> precision, recall, F1 (+ EER do eval).

Saídas em results/<modelo>/: best.pt, history.csv, scores_dev.csv,
scores_eval.csv, metrics.json
"""
import argparse
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from .data import ASVspoofDataset, load_subset
from .features import FeatureExtractor
from .metrics import compute_eer, compute_metrics
from .models import build_model
from .utils import count_params, get_data_root, get_device, load_config, set_seed


def make_loader(df, extractor, cfg, shuffle):
    ds = ASVspoofDataset(df, extractor, cfg["data"]["cache_dir"], get_data_root(cfg))
    return DataLoader(ds, batch_size=cfg["train"]["batch_size"], shuffle=shuffle,
                      num_workers=cfg["train"]["num_workers"],
                      pin_memory=torch.cuda.is_available())


def train_one_epoch(model, loader, criterion, optimizer, device):
    model.train()
    total, n = 0.0, 0
    for x, y, _ in tqdm(loader, desc="train", leave=False):
        x, y = x.to(device), y.to(device)
        optimizer.zero_grad()
        loss = criterion(model(x), y)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
        optimizer.step()
        total += loss.item() * len(y)
        n += len(y)
    return total / n


@torch.no_grad()
def predict(model, loader, device):
    model.eval()
    ids, labels, scores = [], [], []
    for x, y, utt in tqdm(loader, desc="eval", leave=False):
        logits = model(x.to(device))
        scores.append(torch.sigmoid(logits).cpu().numpy())
        labels.append(y.numpy())
        ids.extend(utt)
    return pd.DataFrame({"utt_id": ids, "label": np.concatenate(labels).astype(int),
                         "score": np.concatenate(scores)})


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--config", default="config.yaml")
    ap.add_argument("--epochs", type=int)
    ap.add_argument("--device")
    args = ap.parse_args()

    cfg = load_config(args.config)
    if args.epochs:
        cfg["train"]["epochs"] = args.epochs
    set_seed(cfg["seed"])
    device = get_device(args.device)

    extractor = FeatureExtractor(cfg)
    dfs = {s: load_subset(cfg["data"]["subset_dir"], s) for s in ("train", "dev", "eval")}
    train_loader = make_loader(dfs["train"], extractor, cfg, shuffle=True)
    dev_loader = make_loader(dfs["dev"], extractor, cfg, shuffle=False)
    eval_loader = make_loader(dfs["eval"], extractor, cfg, shuffle=False)

    model = build_model(args.model, extractor.n_features, extractor.n_frames, cfg).to(device)
    n_params = count_params(model)
    print(f"Modelo: {args.model} | params: {n_params:,} | device: {device} | "
          f"features: {extractor.cache_tag()}")

    # pos_weight compensa desbalanceamento caso o subset não seja balanceado
    y = dfs["train"].label.values
    pos_weight = torch.tensor([(y == 0).sum() / max((y == 1).sum(), 1)], device=device)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg["train"]["lr"],
                                 weight_decay=cfg["train"]["weight_decay"])
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min",
                                                           factor=0.5, patience=2)

    out_dir = Path(cfg["output_dir"]) / args.model
    out_dir.mkdir(parents=True, exist_ok=True)
    ckpt = out_dir / "best.pt"

    best_eer, bad_epochs, history = float("inf"), 0, []
    t0 = time.time()
    for epoch in range(1, cfg["train"]["epochs"] + 1):
        loss = train_one_epoch(model, train_loader, criterion, optimizer, device)
        dev = predict(model, dev_loader, device)
        dev_eer, _ = compute_eer(dev.label, dev.score)
        scheduler.step(dev_eer)
        history.append({"epoch": epoch, "train_loss": loss, "dev_eer": dev_eer,
                        "lr": optimizer.param_groups[0]["lr"]})
        flag = ""
        if dev_eer < best_eer:
            best_eer, bad_epochs, flag = dev_eer, 0, " *"
            torch.save(model.state_dict(), ckpt)
        else:
            bad_epochs += 1
        print(f"época {epoch:3d} | loss {loss:.4f} | dev EER {dev_eer * 100:.2f}%{flag}")
        if bad_epochs >= cfg["train"]["patience"]:
            print("Early stopping.")
            break
    train_time = time.time() - t0
    pd.DataFrame(history).to_csv(out_dir / "history.csv", index=False)

    # Avaliação final com o melhor checkpoint
    model.load_state_dict(torch.load(ckpt, map_location=device))
    dev = predict(model, dev_loader, device)
    _, threshold = compute_eer(dev.label, dev.score)
    ev = predict(model, eval_loader, device)
    dev.to_csv(out_dir / "scores_dev.csv", index=False)
    ev.to_csv(out_dir / "scores_eval.csv", index=False)

    results = {
        "model": args.model,
        "n_params": n_params,
        "features": extractor.cache_tag(),
        "epochs_run": len(history),
        "train_time_s": round(train_time, 1),
        "dev": compute_metrics(dev.label, dev.score, threshold),
        "eval": compute_metrics(ev.label, ev.score, threshold),
    }
    with open(out_dir / "metrics.json", "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)

    e = results["eval"]
    print(f"\n[EVAL] precision {e['precision']:.4f} | recall {e['recall']:.4f} | "
          f"F1 {e['f1']:.4f} | EER {e['eer'] * 100:.2f}%")
    print(f"Resultados em {out_dir}/")


if __name__ == "__main__":
    main()
