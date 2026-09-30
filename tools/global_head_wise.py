#!/usr/bin/env python3
"""Trained global head interpolated back toward frozen CLIP (WiSE-FT style).

tools/train_global_head.py's head is residual, h = normalize(x + r(x)), and
starts at r = 0, i.e. at frozen CLIP. Trained on MVTec it helps the targets
that look like MVTec (VisA, Real-IAD) and hurts the far ones (MAD-Real, BMAD).
Interpolating in weight space between the zero-shot and the fine-tuned model
is the standard remedy for that (Wortsman et al., CVPR 2022); for this head it
amounts to scaling the last layer, h_lam = normalize(x + lam * r(x)). No
retraining: the saved heads are reloaded and evaluated for each lam, and lam=1
must reproduce the h_cos written by train_global_head.py.

Output per target dataset: results/globalhead_wise/<tag>/<ds>.npz with
hw<lam>_cos and hw<lam>_cos_flip for every lam, plus img_path.

Usage:
    python tools/global_head_wise.py --runs stab42 stab1 stab2 stab3
"""
import argparse
import os
import sys

import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from train import GlobalImageAdaptor                                # noqa: E402
from train_global_head import TARGETS, cosines, load                # noqa: E402

LAMS = (0.25, 0.5, 0.75)


def main():
	ap = argparse.ArgumentParser()
	ap.add_argument("--emb_root", default="results/globalemb/336")
	ap.add_argument("--head_root", default="results/globalhead")
	ap.add_argument("--out_root", default="results/globalhead_wise")
	ap.add_argument("--runs", nargs="+", default=["stab42", "stab1", "stab2", "stab3"])
	ap.add_argument("--image_adaptor_dropout", type=float, default=0.1)
	args = ap.parse_args()

	device = "cuda" if torch.cuda.is_available() else "cpu"
	for tag in args.runs:
		state = torch.load(os.path.join(args.head_root, tag, "head.pth"), map_location=device)
		D = state["net.1.weight"].shape[1]
		head = GlobalImageAdaptor(embed_dim=D, hidden_dim=D, dropout=args.image_adaptor_dropout).to(device)
		head.load_state_dict(state)
		head.eval()
		os.makedirs(os.path.join(args.out_root, tag), exist_ok=True)
		for ds in TARGETS:
			p = os.path.join(args.emb_root, ds + ".npz")
			if not os.path.isfile(p):
				continue
			d = load(p, device)
			out = {"img_path": d["img_path"]}
			with torch.no_grad():
				for lam in LAMS + (1.0,):
					f = lambda e, lam=lam: F.normalize(e + lam * head.net(e), dim=-1)
					c = cosines(f, d["emb"], d["text"], d["tidx"]).cpu().numpy()
					cf = cosines(f, d["emb_flip"], d["text"], d["tidx"]).cpu().numpy()
					if lam == 1.0:	# must be the head train_global_head.py evaluated
						ref = np.load(os.path.join(args.head_root, tag, ds + ".npz"))["h_cos"]
						print(f"[check] {tag}/{ds}: lam=1 max |diff| vs h_cos = {np.nanmax(np.abs(c - ref)):.2e}")
						continue
					out[f"hw{lam}_cos"], out[f"hw{lam}_cos_flip"] = c, cf
			np.savez_compressed(os.path.join(args.out_root, tag, ds + ".npz"), **out)


if __name__ == "__main__":
	main()
