#!/usr/bin/env python3
"""Cut defect-centred crops out of the datasets for the paper figures.

Each entry names an image and its ground-truth mask; we take the centroid of the
largest defect region, crop a square around it, and write a small PNG into
latex/fig/img/. Nothing here touches the model: it is purely for illustration.

Usage:  python3 tools/make_fig_crops.py
"""

import os

import numpy as np
from PIL import Image, ImageDraw

OUT = "latex/fig/img"
SIDE = 256      # output resolution
CTX = 2.2       # crop side as a multiple of the defect's bounding box
BOX = True      # outline the defect region, so a reader can find it at figure size


def crop(img_path, mask_path, out_name, side=SIDE, ctx=CTX):
	img = Image.open(img_path).convert("RGB")
	m = np.array(Image.open(mask_path).convert("L"))
	ys, xs = np.nonzero(m > 127)
	if len(xs) == 0:
		print(f"  no defect pixels in {mask_path}")
		return None
	cx, cy = int(xs.mean()), int(ys.mean())
	extent = max(xs.max() - xs.min(), ys.max() - ys.min(), 24)
	half = int(min(max(extent * ctx / 2, 60), min(img.size) / 2))
	l = max(0, min(cx - half, img.size[0] - 2 * half))
	t = max(0, min(cy - half, img.size[1] - 2 * half))
	out = img.crop((l, t, l + 2 * half, t + 2 * half)).resize((side, side), Image.LANCZOS)
	if BOX:
		# defect bounding box, mapped into crop coordinates then into output pixels
		sc = side / (2.0 * half)
		x0, x1 = (xs.min() - l) * sc, (xs.max() - l) * sc
		y0, y1 = (ys.min() - t) * sc, (ys.max() - t) * sc
		pad = side * 0.02
		d = ImageDraw.Draw(out)
		d.rectangle([max(1, x0 - pad), max(1, y0 - pad),
					 min(side - 2, x1 + pad), min(side - 2, y1 + pad)],
					outline=(200, 30, 40), width=max(2, side // 90))
	path = os.path.join(OUT, out_name)
	out.save(path)
	frac = 100.0 * (m > 127).sum() / m.size
	print(f"  {out_name}: defect covers {frac:.2f}% of the source image")
	return path


def mvtec(cls, defect, stem):
	return (f"data/mvtec/{cls}/test/{defect}/{stem}.png",
			f"data/mvtec/{cls}/ground_truth/{defect}/{stem}_mask.png")


def mpdd(cls, defect, stem):
	return (f"data/mpdd/{cls}/test/{defect}/{stem}.png",
			f"data/mpdd/{cls}/ground_truth/{defect}/{stem}_mask.png")


# (output name, image, mask). Two groups:
#   src_*  : defect types present in the MVTec training vocabulary
#   new_*  : defect types whose names never appear during training
JOBS = [
	("src_scratch_metalnut.png", *mvtec("metal_nut", "scratch", "000")),
	("src_scratch_wood.png",     *mvtec("wood", "scratch", "000")),
	("src_scratch_capsule.png",  *mvtec("capsule", "scratch", "000")),
	("new_rust_metalplate.png",  *mpdd("metal_plate", "major_rust", "000")),
	("new_flat_tubes.png",       *mpdd("tubes", "flattening", "000")),
	("new_paint_bracket.png",    *mpdd("bracket_white", "defective_painting", "000")),
]

if __name__ == "__main__":
	os.makedirs(OUT, exist_ok=True)
	for name, img, mask in JOBS:
		if not (os.path.exists(img) and os.path.exists(mask)):
			print(f"  SKIP {name}: missing {img if not os.path.exists(img) else mask}")
			continue
		crop(img, mask, name)
