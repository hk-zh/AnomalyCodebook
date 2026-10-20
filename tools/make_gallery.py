#!/usr/bin/env python3
"""Per-product gallery for the supplement: input, ground truth and our map overlaid on
the input, nine anomalous test images per product, grouped by defect type.

Products are those of the MultiADS supplement that are among our targets (VisA pipe
fryum and capsules, MPDD connector and tubes, Real-IAD phone battery and sim card
set), one product drawn at random (--seed) for MAD-Sim, MAD-Real and MVTec-LOCO,
and the single class of each medical set. Within a product, the (up to) three most
frequent defect types share the nine columns, and the images of a type are drawn at
random with the same seed among those with a non-empty mask (not curated). The map
is test.py's (seed-42 checkpoint, entropy-adaptive temperature, sigma=4), as in
tools/make_qualitative.py.

  compute (GPU)  writes one npz with all products
  render (CPU)   one PNG per product, colour range as in the main figure (--vmin_pct)

Usage:
    python tools/make_gallery.py compute --out results/qualitative/gallery/s42.npz
    python tools/make_gallery.py render --inp results/qualitative/gallery/s42.npz \\
        --png_dir latex/fig/img/gallery --vmin_pct 80
"""
import argparse
import csv
import glob
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from make_qualitative import DATASETS, _one, _setup, _target  # noqa: E402
from select_qualitative import pix_score  # noqa: E402

PRODUCTS = [  # (dataset label in DATASETS, class name; None = drawn at random)
	("VisA", "pipe_fryum"), ("VisA", "capsules"),
	("MPDD", "connector"), ("MPDD", "tubes"),
	("Real-IAD", "phone_battery"), ("Real-IAD", "sim_card_set"),
	("MAD-Sim", None), ("MAD-Real", None), ("MVTec-LOCO", None),
	("Brain MRI", None), ("Liver CT", None), ("Retinal OCT", None),
]


def _visa_types(root):
	"""VisA keeps the defect description in each class's image_anno.csv, not in meta.json."""
	out = {}
	for f in glob.glob(os.path.join(root, "*", "image_anno.csv")):
		for row in csv.DictReader(open(f)):
			out[row["image"]] = row["label"]
	return out


def _type(entry, visa):
	t = entry.get("defect_cls", entry.get("specie_name", "")) or visa.get(entry["img_path"], "")
	t = ", ".join(t) if isinstance(t, (list, tuple)) else str(t)
	return t.replace("_", " ").replace(",", ", ").replace("  ", " ").strip()


def compute(args):
	T, torch, device, model, preprocess, tokenizer, linear, codebook, codebooks, targs = _setup(args)
	rec = {k: [] for k in ("image", "mask", "amap", "unc", "label", "cls_name", "img_path")}
	dtype, product = [], []
	rng = np.random.default_rng(args.seed)
	for label, cls in PRODUCTS:
		_, dset, path = next(x for x in DATASETS if x[0] == label)
		data, prompts = _target(T, torch, device, model, preprocess, tokenizer, targs, args, dset, path)
		visa = _visa_types(path) if dset == "visa" else {}
		names = data.get_cls_names()
		cls = cls or names[int(rng.integers(len(names)))]
		groups = {}
		for k, e in enumerate(data.data_all):
			if e.get("cls_name", e.get("product_cls")) == cls and int(e["anomaly"]) == 1:
				groups.setdefault(_type(e, visa), []).append(k)
		types = sorted(groups, key=lambda t: (-len(groups[t]), t))[:args.max_types]
		quota = [args.n // len(types) + (i < args.n % len(types)) for i in range(len(types))]
		for t, q in zip(types, quota):
			taken = 0
			for k in rng.permutation(groups[t]):
				item = data[int(k)]
				if float(item["img_mask_b"].sum()) == 0:
					continue
				_one(item, label, prompts, model, linear, codebook, codebooks, targs, args, rec, T, torch, device)
				dtype.append(t)
				product.append(f"{label} {cls}")
				taken += 1
				if taken == q:
					break
	rec["dtype"], rec["product"] = dtype, product
	rec["pix_score"] = [pix_score(a, m, "ap") for a, m in zip(rec["amap"], rec["mask"])]
	os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
	np.savez_compressed(args.out, **{k: np.asarray(v) for k, v in rec.items()})
	print(f"[saved] {args.out}  {len(product)} images, {len(set(product))} products")


def render(args):
	import matplotlib
	matplotlib.use("Agg")
	import matplotlib.pyplot as plt
	from matplotlib import cm, colors

	d = np.load(args.inp, allow_pickle=False)
	os.makedirs(args.png_dir, exist_ok=True)
	fs = args.fontsize
	for prod in dict.fromkeys(d["product"]):
		idx = [i for i, p in enumerate(d["product"]) if p == prod]
		n = len(idx)
		types = [str(d["dtype"][j]) for j in idx]
		left, right, top, bottom = 0.045, 0.998, 0.90 if any(types) else 0.99, 0.06
		cell = args.width * (right - left) / n
		fig, ax = plt.subplots(3, n, figsize=(args.width, cell * 3 * 1.02 / (top - bottom)), squeeze=False)
		for c, j in enumerate(idx):
			img, m, a = d["image"][j], d["mask"][j], d["amap"][j].astype(float)
			inner = a[12:-12, 12:-12]		# as in make_qualitative.py: the zero-padded border is left out
			lo = np.percentile(inner, args.vmin_pct) if args.vmin_pct > 0 else inner.min()
			heat = cm.jet(colors.Normalize(lo, inner.max(), clip=True)(a))[..., :3]
			ax[0, c].imshow(img)
			ax[1, c].imshow(m, cmap="gray", vmin=0, vmax=1)
			ax[2, c].imshow((1 - args.alpha) * img / 255.0 + args.alpha * heat)
			ax[2, c].text(0.04, 0.96, f"{d['pix_score'][j]:.2f}", transform=ax[2, c].transAxes, ha="left", va="top",
						  fontsize=fs - 1, color="white", bbox=dict(facecolor="black", alpha=0.5, pad=0.6, linewidth=0))
			ax[2, c].set_xlabel(str(c + 1), fontsize=fs, labelpad=1.5)
			for r in range(3):
				ax[r, c].set_xticks([])
				ax[r, c].set_yticks([])
				for s in ax[r, c].spines.values():
					s.set_linewidth(0.3)
		for r, name in enumerate(["Image", "GT", "Ours"]):
			ax[r, 0].set_ylabel(name, fontsize=fs, labelpad=2)
		plt.subplots_adjust(left=left, right=right, top=top, bottom=bottom, wspace=0.03, hspace=0.03)
		s = 0
		while s < n:		# defect type over its columns
			e = s
			while e + 1 < n and types[e + 1] == types[s]:
				e += 1
			if types[s]:
				x0, x1 = ax[0, s].get_position().x0, ax[0, e].get_position().x1
				y = ax[0, s].get_position().y1 + 0.012
				fig.add_artist(plt.Line2D([x0 + 0.004, x1 - 0.004], [y, y], color="black", linewidth=0.5))
				fig.text((x0 + x1) / 2, y + 0.01, types[s], ha="center", va="bottom", fontsize=fs)
			s = e + 1
		png = os.path.join(args.png_dir, prod.lower().replace(" ", "_") + ".png")
		fig.savefig(png, dpi=args.dpi)
		plt.close(fig)
		print(f"[saved] {png}  {n} columns, types {list(dict.fromkeys(types))}, "
			  f"AP {np.round(d['pix_score'][idx], 2).tolist()}")


def main():
	ap = argparse.ArgumentParser()
	ap.add_argument("mode", choices=["compute", "render"])
	ap.add_argument("--out", default="results/qualitative/gallery/s42.npz")
	ap.add_argument("--inp", default="results/qualitative/gallery/s42.npz")
	ap.add_argument("--png_dir", default="latex/fig/img/gallery")
	ap.add_argument("--checkpoint_path", default="./exps/stab42/epoch_1.pth")
	ap.add_argument("--config_path", default="./open_clip/model_configs/ViT-L-14-336.json")
	ap.add_argument("--model", default="ViT-L-14-336")
	ap.add_argument("--image_size", type=int, default=518)
	ap.add_argument("--seed", type=int, default=0, help="seed of the random draws (products and images)")
	ap.add_argument("--n", type=int, default=9, help="images per product")
	ap.add_argument("--max_types", type=int, default=3, help="most frequent defect types shown per product")
	ap.add_argument("--width", type=float, default=6.9)
	ap.add_argument("--fontsize", type=float, default=7)
	ap.add_argument("--alpha", type=float, default=0.5, help="opacity of the map over the input")
	ap.add_argument("--vmin_pct", type=float, default=80.0, help="colour range from this percentile of each map's interior (as in the main figure)")
	ap.add_argument("--dpi", type=int, default=300)
	args = ap.parse_args()
	{"compute": compute, "render": render}[args.mode](args)


if __name__ == "__main__":
	main()
