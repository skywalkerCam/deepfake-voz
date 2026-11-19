"""Cria um mini "ASVspoof 2019 LA" falso para testar o pipeline sem baixar o LA.zip.

bonafide = tons harmônicos com vibrato; spoof = ruído filtrado + tom puro.
Os números obtidos NÃO têm significado científico — é só teste de encanamento.

Uso:
    python -m scripts.make_dummy_dataset --out /tmp/LA_dummy
    # depois aponte data.root para /tmp/LA_dummy (ou use config_dummy.yaml)
"""
import argparse
from pathlib import Path

import numpy as np
import soundfile as sf

SPLITS = {
    "train": ("ASVspoof2019_LA_train", "ASVspoof2019.LA.cm.train.trn.txt", "T",
              ["A01", "A02", "A03"]),
    "dev": ("ASVspoof2019_LA_dev", "ASVspoof2019.LA.cm.dev.trl.txt", "D",
            ["A01", "A02", "A03"]),
    "eval": ("ASVspoof2019_LA_eval", "ASVspoof2019.LA.cm.eval.trl.txt", "E",
             ["A07", "A08", "A09"]),
}
SR = 16000


def bonafide(rng):
    n = int(SR * rng.uniform(1.5, 5.0))
    t = np.arange(n) / SR
    f0 = rng.uniform(100, 250) * (1 + 0.02 * np.sin(2 * np.pi * 5 * t))
    phase = 2 * np.pi * np.cumsum(f0) / SR
    x = sum(np.sin(k * phase) / k for k in range(1, 6))
    return x * 0.1 + rng.normal(0, 0.01, n)


def spoof(rng):
    n = int(SR * rng.uniform(1.5, 5.0))
    t = np.arange(n) / SR
    noise = np.convolve(rng.normal(0, 1, n), np.ones(8) / 8, mode="same")
    return 0.05 * noise + 0.05 * np.sin(2 * np.pi * rng.uniform(100, 250) * t)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="/tmp/LA_dummy")
    ap.add_argument("--n", type=int, default=60, help="amostras por classe por split")
    args = ap.parse_args()

    rng = np.random.RandomState(0)
    root = Path(args.out)
    proto_dir = root / "ASVspoof2019_LA_cm_protocols"
    proto_dir.mkdir(parents=True, exist_ok=True)
    for split, (folder, proto, letter, systems) in SPLITS.items():
        flac = root / folder / "flac"
        flac.mkdir(parents=True, exist_ok=True)
        lines, i = [], 0
        for key in ("bonafide", "spoof"):
            for j in range(args.n):
                utt = f"LA_{letter}_{1000000 + i}"
                i += 1
                wav = bonafide(rng) if key == "bonafide" else spoof(rng)
                sf.write(flac / f"{utt}.flac", wav.astype(np.float32), SR)
                system = "-" if key == "bonafide" else systems[j % len(systems)]
                lines.append(f"LA_{j % 10:04d} {utt} - {system} {key}")
        (proto_dir / proto).write_text("\n".join(lines) + "\n")
        print(f"{split}: {len(lines)} arquivos")
    print(f"Dataset falso em {root}")


if __name__ == "__main__":
    main()
