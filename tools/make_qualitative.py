#!/usr/bin/env python3
"""Qualitative figure: input, ground truth, anomaly map and assignment entropy.

  compute (GPU)  one anomalous test image per dataset, drawn at random with a
                 fixed seed among images with a non-empty mask (not curated); the
                 map is test.py's (seed-42 checkpoint, entropy-adaptive
                 temperature, sigma=4) and U is the normalized entropy of the soft
                 assignment, averaged over the four layers
  render (CPU)   the 4 x N grid as a PNG for the paper, each map min-max
                 normalized per image

Usage:
    python tools/make_qualitative.py compute --out results/qualitative/s42.npz
    python tools/make_qualitative.py render --inp results/qualitative/s42.npz \\
        --png latex/fig/img/qualitative.png
"""
import argparse
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

DATASETS = [  # (column label, loader name, data path)
	("VisA", "visa", "./data/visa"),
	("MPDD", "mpdd", "./data/mpdd"),
	("MAD-Sim", "mad_sim", "./data/MAD-Sim"),
	("MAD-Real", "mad_real", "./data/MAD-Real"),
	("Real-IAD", "real_iad", "./data/Real-IAD"),
	("Brain MRI", "bmad", "./data/BMAD/BraTS2021_slice"),
	("Liver CT", "bmad", "./data/BMAD/Liver"),
	("Retinal OCT", "bmad", "./data/BMAD/RESC"),
	("MVTec-LOCO", "loco", "./data/MMAD/MVTec-LOCO"),
]
MEAN = np.array([0.48145466, 0.4578275, 0.40821073])
STD = np.array([0.26862954, 0.26130258, 0.27577711])


def _setup(args):
	import torch
	import open_clip
	from model import HybridCodebook, LinearLayer
	import test as T

	device = "cuda"
	T.setup_seed(42)
	model, _, preprocess = open_clip.create_model_and_transforms(args.model, args.image_size, pretrained="openai")
	model = model.to(device).eval()
	tokenizer = open_clip.get_tokenizer(args.model)
	ckpt = torch.load(args.checkpoint_path, map_location=device)
	entries = ckpt["hybrid_codebook"]["learnable_entries"]
	codebook = HybridCodebook(semantic_embeddings=None, num_learnable=entries.shape[0], embed_dim=entries.shape[1]).to(device)
	with torch.no_grad():
		codebook.learnable_entries.copy_(entries.to(device))
	codebook.eval()
	codebooks = [codebook] * 4
	cfg = json.load(open(args.config_path))
	linear = LinearLayer(cfg["vision_cfg"]["width"], cfg["embed_dim"], 4, args.model).to(device)
	linear.load_state_dict(ckpt["trainable_linearlayer"])
	linear.eval()

	# test.py's defaults (the paper's setting) for the map
	targs = argparse.Namespace(model=args.model, image_size=args.image_size, assign_temperature=0.1,
							   semantic_bias=0.0, assign_mode="raw", score_blend=0.0, cali=True,
							   temperature=0.01, max_temperature=0.1, split="test", cali_mode="linear",
							   cali_signal="entropy", cali_beta=0.5, expect_temperature=0.02)
	return T, torch, device, model, preprocess, tokenizer, linear, codebook, codebooks, targs


def _target(T, torch, device, model, preprocess, tokenizer, targs, args, dset, path):
	import torch.nn.functional as F
	targs.dataset, targs.data_path = dset, path
	data = T.build_dataset(targs, preprocess, *T.build_target_transforms(args.image_size))
	with torch.inference_mode(), torch.cuda.amp.autocast():
		prompts = T.build_text_prompts(model, data.get_cls_names(), tokenizer, device, dset)
	return data, {k: F.normalize(v.to(device=device, dtype=torch.float32), dim=0) for k, v in prompts.items()}


def compute(args):
	T, torch, device, model, preprocess, tokenizer, linear, codebook, codebooks, targs = _setup(args)
	rec = {k: [] for k in ("image", "mask", "amap", "unc", "label", "cls_name", "img_path")}
	for label, dset, path in DATASETS:
		data, prompts = _target(T, torch, device, model, preprocess, tokenizer, targs, args, dset, path)
		rng = np.random.default_rng(args.seed)
		items = []
		for i in rng.permutation(len(data)):		# the first --per_dataset anomalous images with a mask
			it = data[int(i)]
			if int(it["anomaly"]) == 1 and float(it["img_mask_b"].sum()) > 0:
				items.append(it)
				if len(items) == args.per_dataset:
					break
		for item in items:
			_one(item, label, prompts, model, linear, codebook, codebooks, targs, args, rec, T, torch, device)
	os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
	np.savez_compressed(args.out, **{k: np.asarray(v) for k, v in rec.items()})


def nocali(args):
	"""Our map without the entropy-adaptive temperature (test.py --no-cali: every pixel
	at --temperature 0.01) for the images of --inp, saved next to it as nocali.npz
	in the format of the baseline maps, with the same per-image pixel AP."""
	from select_qualitative import pix_score
	T, torch, device, model, preprocess, tokenizer, linear, codebook, codebooks, targs = _setup(args)
	targs.cali = False
	d = np.load(args.inp, allow_pickle=False)
	labels, paths = list(d["label"]), list(d["img_path"])
	maps = [None] * len(labels)
	for label, dset, path in DATASETS:
		want = [i for i, l in enumerate(labels) if l == label]
		if not want:
			continue
		data, prompts = _target(T, torch, device, model, preprocess, tokenizer, targs, args, dset, path)
		where = {}		# loaders join root (and, for Real-IAD, the class) to the stored path differently
		for k, e in enumerate(data.data_all):
			for q in (e["img_path"], os.path.join(data.root, e["img_path"]),
					  os.path.join(data.root, e.get("cls_name", ""), e["img_path"])):
				where.setdefault(q, k)
		for i in want:
			item = data[where[paths[i]]]
			assert item["img_path"] == paths[i]
			img = item["img"].unsqueeze(0).to(device)
			with torch.inference_mode(), torch.cuda.amp.autocast():
				_, ptoks = model.encode_image(img, [6, 12, 18, 24])
				ptoks = linear(ptoks)
				amap = T.anomaly_map_from_tokens(ptoks, torch.zeros(1, dtype=torch.long, device=device),
												 prompts[item["cls_name"]].unsqueeze(0), codebooks, targs)
			maps[i] = T.gaussian_blur_maps(amap.float(), 4.0)[0].cpu().numpy().astype(np.float16)
	score = np.array([pix_score(m, d["mask"][i], "ap") for i, m in enumerate(maps)])
	out = os.path.join(os.path.dirname(args.inp), "nocali.npz")
	np.savez_compressed(out, amap=np.stack(maps), label=np.array(labels), pix_score=score)
	print(f"[saved] {out}  AP no-cali {np.round(score, 2).tolist()}  cali {np.round(d['pix_score'], 2).tolist() if 'pix_score' in d.files else '-'}")


def _one(item, label, prompts, model, linear, codebook, codebooks, targs, args, rec, T, torch, device):
		img = item["img"].unsqueeze(0).to(device)
		with torch.inference_mode(), torch.cuda.amp.autocast():
			_, ptoks = model.encode_image(img, [6, 12, 18, 24])
			ptoks = linear(ptoks)
			idx = torch.zeros(1, dtype=torch.long, device=device)
			text = prompts[item["cls_name"]].unsqueeze(0)
			amap = T.anomaly_map_from_tokens(ptoks, idx, text, codebooks, targs)
			unc = torch.stack([
				T.build_calibration_map(
					codebook(T.extract_patch_tokens_for_layer(p, args.model).float(),
							 assign_temperature=0.1)["assign_probs"], args.image_size)
				for p in ptoks]).mean(0)
		amap = T.gaussian_blur_maps(amap.float(), 4.0)
		rgb = (item["img"].permute(1, 2, 0).numpy() * STD + MEAN).clip(0, 1)
		rec["image"].append((rgb * 255).astype(np.uint8))
		rec["mask"].append(item["img_mask_b"].squeeze().numpy() > 0.5)
		rec["amap"].append(amap[0].cpu().numpy().astype(np.float16))
		rec["unc"].append(unc[0].float().cpu().numpy().astype(np.float16))
		rec["label"].append(label)
		rec["cls_name"].append(item["cls_name"])
		rec["img_path"].append(item["img_path"])
		print(f"{label:12s} {item['cls_name']:16s} {item['img_path']}  map {float(amap.min()):.3f}-{float(amap.max()):.3f}")


METHODS = [("anomalyclip", "AnomalyCLIP"), ("dapo", "DAPO"), ("multiads", "MultiADS"),
		   ("nocali", "Ours,\nfixed $T$")]
# the compact (main-text) layout: the datasets of Table 1 plus two transfer targets,
# fixed before any baseline map was rendered
COMPACT = ["VisA", "MPDD", "Real-IAD", "Retinal OCT", "MVTec-LOCO"]


def render(args):
	import matplotlib
	matplotlib.use("Agg")
	import matplotlib.pyplot as plt
	from matplotlib import colors

	d = np.load(args.inp, allow_pickle=False)
	labels = list(d["label"])
	base, aurocs = {}, {}
	for key, name in METHODS:		# baseline maps from tools/qual_baselines.py, if computed
		if args.methods and key not in args.methods:
			continue
		f = os.path.join(os.path.dirname(args.inp), key + ".npz")
		if os.path.isfile(f):
			b = np.load(f)
			assert list(b["label"]) == labels, f"example order differs: {f}"
			base[name] = b["amap"]
			if "pix_score" in b.files:
				aurocs[name] = b["pix_score"]
	if "pix_score" in d.files:
		aurocs["Ours"] = d["pix_score"]
	pick = args.columns or (COMPACT if args.layout == "compact" and set(COMPACT) <= set(labels) else labels)
	cols = [labels.index(c) for c in pick]
	maps = [(name, base[name]) for _, name in METHODS if name in base] + [("Ours", d["amap"])]
	if not all(n in aurocs for n, _ in maps):		# insets on every map row or on none
		aurocs = {}
	if args.layout == "compact":	# GT on the input, no entropy row
		rows = [("Image + GT", "img_gt")] + [(n, ("map", m)) for n, m in maps]
	else:
		rows = [("Image", "img"), ("Ground truth", "img_gt")] + [(n, ("map", m)) for n, m in maps] + [("Ours: entropy $U$", "unc")]
	n = len(cols)
	fs = args.fontsize or (6 if args.layout == "compact" else 7)
	left = 0.1 if args.layout == "compact" else 0.04
	right, top, bottom = 0.995, 0.95, 0.005
	cell = args.width * (right - left) / n			# square cells: height follows from the width
	fig, ax = plt.subplots(len(rows), n, figsize=(args.width, cell * len(rows) * 1.03 / (top - bottom)), squeeze=False)

	def norm(a, b=12):
		# colour range from the interior: test.py's smoothing zero-pads, which lowers
		# a ~3 sigma border band of our map; the same rule is applied to every method
		a = a.astype(float)
		inner = a[b:-b, b:-b]
		lo = np.percentile(inner, args.vmin_pct) if args.vmin_pct > 0 else inner.min()
		return a, colors.Normalize(lo, inner.max(), clip=True)

	for c, j in enumerate(cols):
		img, m = d["image"][j], d["mask"][j]
		for r, (name, kind) in enumerate(rows):
			a_ = ax[r, c]
			if kind in ("img", "img_gt"):
				a_.imshow(img)
				if kind == "img_gt":
					a_.contour(m, levels=[0.5], colors="red", linewidths=0.6)
			elif kind == "unc":
				u, nu = norm(d["unc"][j])
				a_.imshow(u, cmap="magma", norm=nu)
			else:		# anomaly map, min-max normalized per image, GT outlined (magenta: not in jet)
				a, na = norm(kind[1][j])
				a_.imshow(a, cmap="jet", norm=na)
				if args.gt_width > 0:
					a_.contour(m, levels=[0.5], colors=args.gt_color, linewidths=args.gt_width)
				if name in aurocs:		# per-image pixel AP, from tools/select_qualitative.py
					a_.text(0.04, 0.96, f"{aurocs[name][j]:.2f}", transform=a_.transAxes, ha="left", va="top",
							fontsize=fs - 1, color="white", bbox=dict(facecolor="black", alpha=0.5, pad=0.8, linewidth=0))
			a_.set_xticks([])
			a_.set_yticks([])
			for s in a_.spines.values():
				s.set_linewidth(0.3)
		ax[0, c].set_title(labels[j], fontsize=fs, pad=2)
	for r, (name, _) in enumerate(rows):
		ax[r, 0].set_ylabel(name, fontsize=fs)
	plt.subplots_adjust(left=left, right=right, top=top, bottom=bottom, wspace=0.03, hspace=0.03)
	os.makedirs(os.path.dirname(os.path.abspath(args.png)), exist_ok=True)
	fig.savefig(args.png, dpi=args.dpi)
	print(f"[saved] {args.png}  ({args.layout}, {n} columns, {len(rows)} rows, baselines: {sorted(base)})")


def main():
	ap = argparse.ArgumentParser()
	ap.add_argument("mode", choices=["compute", "nocali", "render"])
	ap.add_argument("--out", default="results/qualitative/s42.npz")
	ap.add_argument("--inp", default="results/qualitative/s42.npz")
	ap.add_argument("--png", default="latex/fig/img/qualitative.png")
	ap.add_argument("--checkpoint_path", default="./exps/stab42/epoch_1.pth")
	ap.add_argument("--config_path", default="./open_clip/model_configs/ViT-L-14-336.json")
	ap.add_argument("--model", default="ViT-L-14-336")
	ap.add_argument("--image_size", type=int, default=518)
	ap.add_argument("--seed", type=int, default=0, help="seed of the random draw of examples")
	ap.add_argument("--per_dataset", type=int, default=1, help="anomalous images drawn per dataset (a pool for tools/select_qualitative.py when > 1)")
	ap.add_argument("--width", type=float, default=6.9, help="figure width in inches (6.9 = text width, 3.3 = one column)")
	ap.add_argument("--layout", choices=["full", "compact"], default="full")
	ap.add_argument("--columns", nargs="+", default=None, help="dataset labels to show, in order (default: all, or the compact set)")
	ap.add_argument("--methods", nargs="+", default=None, choices=[k for k, _ in METHODS], help="baselines to show (default: all computed)")
	ap.add_argument("--fontsize", type=float, default=None, help="label size in pt at --width (default 6 compact, 7 full)")
	ap.add_argument("--gt_color", default="magenta", help="GT outline on the maps")
	ap.add_argument("--gt_width", type=float, default=1.5, help="GT outline width on the maps (pt at --width); 0 draws none")
	ap.add_argument("--vmin_pct", type=float, default=0.0, help="colour range starts at this percentile of each map's interior instead of its minimum (same rule for every method; AP is unaffected)")
	ap.add_argument("--dpi", type=int, default=300)
	args = ap.parse_args()
	{"compute": compute, "nocali": nocali, "render": render}[args.mode](args)


if __name__ == "__main__":
	main()
