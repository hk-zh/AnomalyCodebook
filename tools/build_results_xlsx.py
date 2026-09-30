#!/usr/bin/env python
"""Assemble all AnomalyCodebook results tried this cycle into one .xlsx.

Sweep numbers are read straight from results/sweep_alpha0/**/log.txt and the
LOCO JSON so the sheet cannot drift from the logs; paper / BMAD / DAPO numbers
are the verified values already on the slide deck.
"""
import json
import os
import re

import xlsxwriter

ROOT = "/home/zho1rng/AnomalyCodebook"
OUT = os.path.join(ROOT, "results", "AnomalyCodebook_results_summary.xlsx")


def meanrow(fp):
    """Return (pixel_auroc, aupro, image_auroc, image_ap) from a test log."""
    row = None
    if not os.path.exists(fp):
        return None
    for line in open(fp):
        if line.startswith("| mean") or line.startswith("| resc"):
            row = line
    if not row:
        return None
    n = re.findall(r"[-\d.]+", row.split("|", 2)[2])
    try:
        return float(n[0]), float(n[3]), float(n[4]), float(n[6])
    except Exception:
        return None


# ---------------------------------------------------------------- data ------
MAIN = [  # Dataset, Method, PixelAUROC, AUPRO, ImageAUROC, ImageAP
    ("VisA", "WinCLIP", 79.6, 56.8, 78.1, 81.2), ("VisA", "April-GAN", 94.2, 86.8, 78.0, 81.4),
    ("VisA", "AnomalyCLIP", 95.5, 87.0, 82.1, 85.4), ("VisA", "AdaCLIP", 95.0, None, 75.4, 79.3),
    ("VisA", "MultiADS", 95.0, 89.7, 83.6, 86.9), ("VisA", "Ours (a=0)", 94.6, 89.6, 83.0, 86.2),
    ("MPDD", "WinCLIP", 76.4, 48.9, 63.6, 69.9), ("MPDD", "April-GAN", 94.1, 83.2, 73.0, 80.2),
    ("MPDD", "AnomalyCLIP", 96.5, 88.7, 77.0, 82.0), ("MPDD", "AdaCLIP", 96.3, 66.3, 75.0, None),
    ("MPDD", "MultiADS-F", 96.3, 89.5, 79.7, 80.5), ("MPDD", "Ours (a=0, rho=0.1)", 96.5, 90.7, 75.6, 77.3),
    ("MAD-Sim", "WinCLIP", 77.6, 55.8, 54.3, 90.2), ("MAD-Sim", "April-GAN", 80.4, 61.5, 56.0, 91.0),
    ("MAD-Sim", "AnomalyCLIP", 77.9, 40.1, 54.6, 90.9), ("MAD-Sim", "MultiADS", 88.0, 74.2, 57.1, 94.4),
    ("MAD-Sim", "Ours (a=0)", 88.3, 77.9, 59.7, 92.1),
    ("MAD-Real", "WinCLIP", 60.5, 26.9, 64.1, 87.6), ("MAD-Real", "April-GAN", 88.2, 69.5, 62.9, 87.7),
    ("MAD-Real", "AnomalyCLIP", 88.3, 65.1, 66.8, 90.0), ("MAD-Real", "MultiADS-F", 90.7, 75.2, 78.5, 92.9),
    ("MAD-Real", "Ours (a=0)", 88.0, 71.7, 68.8, 89.9),
    ("Real-IAD", "WinCLIP", 87.1, 59.9, 75.0, 72.3), ("Real-IAD", "April-GAN", 96.0, 86.8, 75.7, 73.5),
    ("Real-IAD", "AnomalyCLIP", 96.2, 85.7, 78.4, 76.7), ("Real-IAD", "AdaCLIP", 95.3, None, 70.1, 68.5),
    ("Real-IAD", "MultiADS", 96.6, 87.1, 78.7, 79.1), ("Real-IAD", "Ours (a=0)", 97.1, 91.7, 81.1, 81.1),
]

BMAD = [  # Domain, Method, PixelAUROC, ImageAUROC, ImageAP, ImageF1
    ("brain", "Ours (a=0)", 94.3, 73.0, 92.2, 90.7), ("brain", "DAPO", 95.8, 58.2, 86.4, 90.7),
    ("liver", "Ours (a=0)", 96.3, 64.9, 56.6, 65.9), ("liver", "DAPO", 97.5, 55.5, 47.9, 62.6),
    ("resc", "Ours (a=0, rho=0.05)", 90.7, 81.0, 80.9, 71.7), ("resc", "Ours (a=0, rho=0.5*)", 90.7, 83.1, 83.1, None),
    ("resc", "DAPO", 93.1, 80.8, 71.3, 72.2),
    ("macro (uniform rho)", "Ours (a=0)", 93.8, 73.0, 76.6, 76.1),
    ("macro (per-domain rho*)", "Ours (a=0)", 93.8, 73.7, 77.3, None),
    ("macro", "DAPO", 95.5, 64.8, 68.5, 75.2),
]

TOPK = {  # dataset -> (data-subdir, [rhos])
    "VisA": ("topk/visa", ["0.001", "0.005", "0.01", "0.02", "0.05", "0.1", "0.2"]),
    "MAD-Sim": ("topk/mad_sim", ["0.001", "0.005", "0.01", "0.02", "0.05", "0.1", "0.2"]),
    "MAD-Real": ("topk/mad_real", ["0.001", "0.005", "0.01", "0.02", "0.05", "0.1", "0.2"]),
    "Real-IAD": ("topk/real_iad", ["0.001", "0.02", "0.05", "0.1"]),
    "BMAD-resc": ("resc_topk", ["0.001", "0.02", "0.05", "0.1", "0.2", "0.5"]),
}
CALI = {  # dataset -> (subdir, baseline image AUROC, [configs])
    "VisA": ("cali/visa", 83.0), "MPDD": ("cali/mpdd", 75.6), "MAD-Real": ("cali/mad_real", 69.8),
}
CALI_CFG = ["global", "aT0.05", "aT0.2", "aT0.5", "maxT0.2", "nocali"]

OVERVIEW = [
    ("VisA", "top-k + cali", "83.0", "83.1", "maxT=0.2", "+0.1", "default optimal"),
    ("MAD-Sim", "top-k", "59.7", "59.7", "rho=0.001", "-", "default optimal"),
    ("MAD-Real", "top-k + cali", "69.8", "70.6", "rho=0.05, aT=0.2", "+0.8", "small gain"),
    ("Real-IAD", "top-k", "81.1", "81.1", "rho=0.001", "-", "default optimal"),
    ("BMAD-resc", "top-k", "81.0", "83.1", "rho=0.5 (val-selected)", "+2.1", "REAL WIN"),
    ("MPDD", "top-k + cali", "75.6", "75.6", "default cali", "-", "default optimal"),
]

# LOCO from the evaluator JSON
loco = {}
lf = os.path.join(ROOT, "results/loco_spro/auspro_ours_vs_dapo.txt")
for line in open(lf):
    if line.strip().startswith("JSON "):
        d = json.loads(line.split("JSON ", 1)[1])
        loco[d["tag"]] = d


# --------------------------------------------------------------- write ------
os.makedirs(os.path.dirname(OUT), exist_ok=True)
wb = xlsxwriter.Workbook(OUT, {"nan_inf_to_errors": True})
DARK, ACCENT, HIL, WARN = "#0B3D5C", "#0072BC", "#E4F0F7", "#FBE7E2"
f_title = wb.add_format({"bold": True, "font_size": 15, "font_color": DARK})
f_note = wb.add_format({"italic": True, "font_size": 10, "font_color": "#666666", "text_wrap": True})
f_hdr = wb.add_format({"bold": True, "font_color": "white", "bg_color": DARK, "border": 1, "align": "center", "valign": "vcenter", "text_wrap": True})
f_txt = wb.add_format({"border": 1})
f_txtb = wb.add_format({"border": 1, "bold": True})
f_num = wb.add_format({"border": 1, "num_format": "0.0", "align": "center"})
f_ours = wb.add_format({"border": 1, "bold": True, "font_color": ACCENT, "bg_color": HIL})
f_oursn = wb.add_format({"border": 1, "bold": True, "font_color": ACCENT, "bg_color": HIL, "num_format": "0.0", "align": "center"})
f_best = wb.add_format({"border": 1, "bold": True, "bg_color": "#FFF3C4", "num_format": "0.0", "align": "center"})
f_win = wb.add_format({"border": 1, "bold": True, "bg_color": HIL})
f_warn = wb.add_format({"border": 1, "bg_color": WARN})


def num(ws, r, c, v, fmt):
    if v is None:
        ws.write(r, c, "-", f_txt)
    else:
        ws.write_number(r, c, v, fmt)


# ---- Overview
ws = wb.add_worksheet("Overview")
ws.set_column(0, 0, 16); ws.set_column(1, 1, 14); ws.set_column(2, 4, 20); ws.set_column(5, 5, 8); ws.set_column(6, 6, 18)
ws.write(0, 0, "AnomalyCodebook - results summary (alpha=0 inference cycle, 2026-08-21)", f_title)
ws.merge_range(1, 0, 1, 6, "All eval is zero-shot from mvtec_zeroshot/epoch_1. Image AUROC is the metric the sweeps move "
               "(pixel AUROC / AUPRO are top-k invariant). * = validation-selected per-domain rho.", f_note)
hdr = ["Dataset", "Swept", "Default", "Best found", "Winning config", "Delta", "Verdict"]
for c, h in enumerate(hdr):
    ws.write(3, c, h, f_hdr)
for i, row in enumerate(OVERVIEW, start=4):
    win = row[-1] == "REAL WIN"
    for c, v in enumerate(row):
        ws.write(i, c, v, f_win if win else (f_txtb if c == 0 else f_txt))
ws.freeze_panes(4, 0)


# ---- Main comparison
ws = wb.add_worksheet("Main comparison (paper)")
ws.set_column(0, 0, 12); ws.set_column(1, 1, 20); ws.set_column(2, 5, 13)
ws.write(0, 0, "Zero-shot AD vs prior methods (MVTec-trained -> target). Ours = alpha=0, epoch 1.", f_title)
hdr = ["Dataset", "Method", "Pixel AUROC", "AUPRO", "Image AUROC", "Image AP"]
for c, h in enumerate(hdr):
    ws.write(2, c, h, f_hdr)
for i, (ds, m, px, pro, ia, ap) in enumerate(MAIN, start=3):
    ours = m.startswith("Ours")
    ws.write(i, 0, ds, f_txtb); ws.write(i, 1, m, f_ours if ours else f_txt)
    for c, v in zip(range(2, 6), (px, pro, ia, ap)):
        num(ws, i, c, v, f_oursn if ours else f_num)
ws.freeze_panes(3, 0)


# ---- BMAD
ws = wb.add_worksheet("BMAD vs DAPO")
ws.set_column(0, 0, 24); ws.set_column(1, 1, 22); ws.set_column(2, 5, 13)
ws.write(0, 0, "BMAD - Ours (alpha=0) vs DAPO, per medical domain.", f_title)
ws.merge_range(1, 0, 1, 5, "resc rho=0.5 is validation-selected (val image AUROC climbs monotonically to 82.9 at "
               "rho=0.5); it lifts resc test image AUROC 81.0->83.1 and BMAD macro 73.0->73.7.", f_note)
hdr = ["Domain", "Method", "Pixel AUROC", "Image AUROC", "Image AP", "Image F1"]
for c, h in enumerate(hdr):
    ws.write(3, c, h, f_hdr)
for i, (dom, m, px, ia, ap, f1) in enumerate(BMAD, start=4):
    ours = m.startswith("Ours")
    ws.write(i, 0, dom, f_txtb); ws.write(i, 1, m, f_ours if ours else f_txt)
    for c, v in zip(range(2, 6), (px, ia, ap, f1)):
        num(ws, i, c, v, f_oursn if ours else f_num)
ws.freeze_panes(4, 0)


# ---- LOCO
ws = wb.add_worksheet("LOCO AU-sPRO")
ws.set_column(0, 0, 22); ws.set_column(1, 7, 12)
ws.write(0, 0, "MVTec-LOCO - same evaluator (tools/spro.py) on both dumps.", f_title)
ws.merge_range(1, 0, 1, 7, "AU-sPRO = saturated Per-Region Overlap, area under the sPRO-vs-FPR curve up to the FPR limit. "
               "@0.05 is the official headline. L/S = logical / structural anomaly split.", f_note)
hdr = ["Metric", "Ours mean", "Ours L", "Ours S", "DAPO mean", "DAPO L", "DAPO S", "Delta (mean)"]
for c, h in enumerate(hdr):
    ws.write(3, c, h, f_hdr)
metrics = [("AU-sPRO @0.05 (headline)", "au_spro", "0.05"), ("AU-sPRO @0.3", "au_spro", "0.3"),
           ("Image AUROC", "image_auroc", None)]
for i, (name, key, lim) in enumerate(metrics, start=4):
    o = loco["ours"][key]; d = loco["dapo"][key]
    if lim:
        o = o[lim]; d = d[lim]
    ov = [o["mean"] * 100, o["logical"] * 100, o["structural"] * 100]
    dv = [d["mean"] * 100, d["logical"] * 100, d["structural"] * 100]
    ws.write(i, 0, name, f_txtb)
    for c, v in enumerate(ov, start=1):
        ws.write_number(i, c, v, f_oursn if c == 1 else f_num)
    for c, v in enumerate(dv, start=4):
        ws.write_number(i, c, v, f_num)
    ws.write_number(i, 7, ov[0] - dv[0], f_best)
ws.freeze_panes(4, 0)


# ---- Top-k sweep
ws = wb.add_worksheet("Top-k sweep")
ws.set_column(0, 0, 12); ws.set_column(1, 1, 8); ws.set_column(2, 5, 13)
ws.write(0, 0, "Per-dataset top-k (rho) sweep, alpha=0. Best image AUROC per dataset highlighted.", f_title)
hdr = ["Dataset", "rho", "Pixel AUROC", "AUPRO", "Image AUROC", "Image AP"]
for c, h in enumerate(hdr):
    ws.write(2, c, h, f_hdr)
r = 3
for ds, (sub, rhos) in TOPK.items():
    rows = []
    for rho in rhos:
        m = meanrow(os.path.join(ROOT, "results/sweep_alpha0", sub, f"topk{rho}", "log.txt"))
        if m:
            rows.append((rho, m))
    best = max((ia for _, (_, _, ia, _) in rows), default=None)
    for rho, (px, pro, ia, ap) in rows:
        isbest = ia == best
        ws.write(r, 0, ds, f_txtb); ws.write(r, 1, rho, f_txt)
        num(ws, r, 2, px, f_num); num(ws, r, 3, pro, f_num)
        num(ws, r, 4, ia, f_best if isbest else f_num); num(ws, r, 5, ap, f_num)
        r += 1
ws.freeze_panes(3, 0)


# ---- Calibration sweep
ws = wb.add_worksheet("Calibration sweep")
ws.set_column(0, 0, 12); ws.set_column(1, 1, 10); ws.set_column(2, 6, 13)
ws.write(0, 0, "Calibration / global-branch sweep, alpha=0 (targets the image-AUROC gap).", f_title)
ws.merge_range(1, 0, 1, 6, "Configs: global = global image branch; aT = assign_temperature; maxT = max_temperature; "
               "nocali = entropy calibration off. Baseline = default cali at the dataset's default rho.", f_note)
hdr = ["Dataset", "Config", "Pixel AUROC", "AUPRO", "Image AUROC", "Image AP", "Delta vs base"]
for c, h in enumerate(hdr):
    ws.write(3, c, h, f_hdr)
r = 4
for ds, (sub, base) in CALI.items():
    for cfg in CALI_CFG:
        m = meanrow(os.path.join(ROOT, "results/sweep_alpha0", sub, cfg, "log.txt"))
        if not m:
            continue
        px, pro, ia, ap = m
        d = ia - base
        better = d > 0.15
        ws.write(r, 0, ds, f_txtb); ws.write(r, 1, cfg, f_txt)
        num(ws, r, 2, px, f_num); num(ws, r, 3, pro, f_num)
        num(ws, r, 4, ia, f_best if better else f_num); num(ws, r, 5, ap, f_num)
        ws.write_number(r, 6, d, f_win if better else (f_warn if d < -1 else f_num))
        r += 1
ws.freeze_panes(4, 0)

wb.close()
print("wrote", OUT)
