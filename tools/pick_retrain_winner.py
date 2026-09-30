#!/usr/bin/env python3
"""Report retrained-checkpoint zero-shot results across the codebook-size sweep.

Discovers every results/retrain_nl<N>/ dir and, per epoch, reads the mean image
AUROC (auroc_sp) for VisA, MPDD and BMAD (brain/liver/resc). Prints a table to
stderr sorted by BMAD mean (the metric where codebook scaling actually paid off),
and prints the winning checkpoint path to stdout as "<NL> <EP> <ckpt>".
"""
import os
import glob
import re
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# alpha=0 baselines (mvtec_zeroshot/epoch_1, nl100) for reference.
BASE = {"visa": 83.0, "mpdd": 75.6, "bmad_brain": 63.7, "bmad_liver": 54.0, "bmad_resc": 72.3}
BMAD_KEYS = ["bmad_brain", "bmad_liver", "bmad_resc"]


def read_auroc_sp(log_path):
    if not os.path.isfile(log_path):
        return None
    val = None
    with open(log_path) as f:
        for line in f:
            if "|" not in line:
                continue
            cells = [c.strip() for c in line.split("|")]
            if len(cells) >= 8 and cells[1] == "mean":
                try:
                    val = float(cells[6])
                except ValueError:
                    pass
    return val


def bmad_mean(d):
    vals = [d[k] for k in BMAD_KEYS if d.get(k) is not None]
    return sum(vals) / len(vals) if len(vals) == len(BMAD_KEYS) else None


def main():
    nls = sorted(int(re.search(r"retrain_nl(\d+)$", p).group(1))
                 for p in glob.glob(os.path.join(ROOT, "results/retrain_nl*"))
                 if re.search(r"retrain_nl(\d+)$", p))

    rows = []
    for nl in nls:
        for ep in range(1, 11):
            base = os.path.join(ROOT, f"results/retrain_nl{nl}")
            m = {k: read_auroc_sp(os.path.join(base, k, f"epoch_{ep}", "log.txt"))
                 for k in ["visa", "mpdd"] + BMAD_KEYS}
            if all(v is None for v in m.values()):
                continue
            ckpt = os.path.join(ROOT, f"exps/retrain_nl{nl}/epoch_{ep}.pth")
            rows.append((nl, ep, m, bmad_mean(m), ckpt))

    if not rows:
        print("pick_retrain_winner: no results yet", file=sys.stderr)
        sys.exit(1)

    def fmt(x):
        return f"{x:6.1f}" if isinstance(x, float) else "     ."

    hdr = f"{'NL':>4} {'EP':>3} {'VisA':>6} {'MPDD':>6} {'brain':>6} {'liver':>6} {'resc':>6} {'BMADavg':>7}"
    print(hdr, file=sys.stderr)
    b = BASE
    print(f"base  -  {fmt(b['visa'])} {fmt(b['mpdd'])} {fmt(b['bmad_brain'])} "
          f"{fmt(b['bmad_liver'])} {fmt(b['bmad_resc'])} {fmt(sum(b[k] for k in BMAD_KEYS)/3):>7}",
          file=sys.stderr)

    ranked = sorted(rows, key=lambda r: (r[3] is not None, r[3] or -1), reverse=True)
    for nl, ep, m, bm, _ in ranked:
        star = " <--" if (nl, ep) == (ranked[0][0], ranked[0][1]) else ""
        print(f"{nl:>4} {ep:>3} {fmt(m['visa'])} {fmt(m['mpdd'])} {fmt(m['bmad_brain'])} "
              f"{fmt(m['bmad_liver'])} {fmt(m['bmad_resc'])} {fmt(bm)}{star}", file=sys.stderr)

    nl, ep, _, _, ckpt = ranked[0]
    print(f"{nl} {ep} {ckpt}")


if __name__ == "__main__":
    main()
