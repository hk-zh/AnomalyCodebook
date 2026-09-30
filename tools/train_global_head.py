#!/usr/bin/env python3
"""Trained global score that bypasses the codebook.

tools/image_score_fusion.py found that frozen CLIP's global zero-shot score
(G336f) complements the map read-out, while the model's own global branch
(image adaptor -> shared codebook) collapses to one prototype. This trains the
same GlobalImageAdaptor as train.py, with the same image-level BCE, optimizer,
learning rate, batch size, temperature and single epoch on MVTec, but without
the codebook: the adapted embedding is compared with the text anchors directly.
Nothing in the pixel path is touched, so the stab* checkpoints stay as they are.

Two choices differ from train.py's branch:
  * the last layer starts at zero, so the head starts exactly at frozen G and
    moves only as far as the MVTec labels push it;
  * features are precomputed (tools/global_native_scores.py --save_emb, 336 px),
    so augmentation is a random horizontal flip instead of the mosaic.

One head per seed, written next to the matching checkpoint's tag so the fusion
pairs head seed s with map seed s. Output per target dataset:
results/globalhead/<tag>/<ds>.npz with h_cos, h_cos_flip, img_path.

Usage:
    python tools/train_global_head.py --emb_root results/globalemb/336 \\
        --runs stab42:42 stab1:1 stab2:2 stab3:3
"""
import argparse
import os
import sys

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics import roc_auc_score

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from train import GlobalImageAdaptor, setup_seed                    # noqa: E402

TARGETS = ["visa", "mpdd", "mad_sim", "mad_real", "real_iad",
		   "bmad_brain", "bmad_liver", "bmad_resc", "loco"]


def load(path, device):
	d = np.load(path, allow_pickle=False)
	cls_idx = {c: i for i, c in enumerate(d["text_cls"])}
	return {
		"emb": torch.from_numpy(d["emb"].astype(np.float32)).to(device),
		"emb_flip": torch.from_numpy(d["emb_flip"].astype(np.float32)).to(device),
		"text": torch.from_numpy(d["text_emb"].astype(np.float32)).to(device),	# [K, D, C] NaN-padded
		"tidx": torch.tensor([cls_idx[c] for c in d["cls_name"]], device=device),
		"label": torch.from_numpy(d["label"].astype(np.float32)).to(device),
		"cls_name": d["cls_name"],
		"img_path": d["img_path"],
	}


def cosines(head, emb, text, tidx):
	"""-> [N, C] cosine of the adapted embedding with its class's anchors (NaN past C)."""
	h = head(emb)												# normalized
	return torch.bmm(h.unsqueeze(1), text[tidx]).squeeze(1)


def binary_logit(cos, temperature):
	"""train.py's image-level logit: logsumexp over anomaly anchors minus normal."""
	z = torch.where(torch.isnan(cos), torch.full_like(cos, -float("inf")), cos / temperature)
	return torch.logsumexp(z[:, 1:], dim=1) - z[:, 0]


def per_class_auroc(d, score):
	lab = d["label"].cpu().numpy().astype(int)
	s = score.cpu().numpy()
	aucs = []
	for c in sorted(set(d["cls_name"])):
		m = d["cls_name"] == c
		if len(set(lab[m])) == 2:
			aucs.append(roc_auc_score(lab[m], s[m]))
	return 100 * np.mean(aucs)


def main():
	ap = argparse.ArgumentParser()
	ap.add_argument("--emb_root", default="results/globalemb/336")
	ap.add_argument("--out_root", default="results/globalhead")
	ap.add_argument("--runs", nargs="+", default=["stab42:42", "stab1:1", "stab2:2", "stab3:3"],
					help="tag:seed pairs; the tag names the map checkpoint the head is paired with")
	# train.py defaults
	ap.add_argument("--epoch", type=int, default=1)
	ap.add_argument("--learning_rate", type=float, default=0.001)
	ap.add_argument("--batch_size", type=int, default=4)
	ap.add_argument("--image_temperature", type=float, default=0.1)
	ap.add_argument("--image_adaptor_dropout", type=float, default=0.1)
	ap.add_argument("--grad_clip", type=float, default=1.0)
	args = ap.parse_args()

	device = "cuda" if torch.cuda.is_available() else "cpu"
	tr = load(os.path.join(args.emb_root, "mvtec.npz"), device)
	N, D = tr["emb"].shape
	targets = {ds: load(os.path.join(args.emb_root, ds + ".npz"), device)
			   for ds in TARGETS if os.path.isfile(os.path.join(args.emb_root, ds + ".npz"))}
	print(f"MVTec train: {N} images ({int(tr['label'].sum())} anomalous); targets: {list(targets)}")

	for run in args.runs:
		tag, seed = run.split(":")
		setup_seed(int(seed))
		head = GlobalImageAdaptor(embed_dim=D, hidden_dim=D, dropout=args.image_adaptor_dropout).to(device)
		torch.nn.init.zeros_(head.net[-1].weight)
		torch.nn.init.zeros_(head.net[-1].bias)
		opt = torch.optim.Adam(head.parameters(), lr=args.learning_rate, betas=(0.5, 0.999))

		def evaluate(d):
			head.eval()
			with torch.no_grad():
				c = cosines(head, d["emb"], d["text"], d["tidx"])
				cf = cosines(head, d["emb_flip"], d["text"], d["tidx"])
			return c, cf

		c0, _ = evaluate(tr)
		auc0 = per_class_auroc(tr, binary_logit(c0, args.image_temperature))

		g = torch.Generator().manual_seed(int(seed))
		losses = []
		for _ in range(args.epoch):
			head.train()
			for idx in torch.randperm(N, generator=g).split(args.batch_size):
				idx = idx.to(device)
				flip = torch.rand(len(idx), generator=g).to(device) < 0.5
				x = torch.where(flip[:, None], tr["emb_flip"][idx], tr["emb"][idx])
				logit = binary_logit(cosines(head, x, tr["text"], tr["tidx"][idx]), args.image_temperature)
				loss = F.binary_cross_entropy_with_logits(logit, tr["label"][idx])
				opt.zero_grad(set_to_none=True)
				loss.backward()
				if args.grad_clip > 0:
					torch.nn.utils.clip_grad_norm_(head.parameters(), args.grad_clip)
				opt.step()
				losses.append(loss.item())

		c1, _ = evaluate(tr)
		auc1 = per_class_auroc(tr, binary_logit(c1, args.image_temperature))
		print(f"[{tag} seed {seed}] {len(losses)} steps, BCE {np.mean(losses[:50]):.3f} -> "
			  f"{np.mean(losses[-50:]):.3f}; MVTec train AUROC {auc0:.1f} -> {auc1:.1f}")

		os.makedirs(os.path.join(args.out_root, tag), exist_ok=True)
		for ds, d in targets.items():
			c, cf = evaluate(d)
			s0 = per_class_auroc(d, binary_logit(cosines(
				lambda e: F.normalize(e, dim=-1), d["emb"], d["text"], d["tidx"]), args.image_temperature))
			s1 = per_class_auroc(d, binary_logit(c, args.image_temperature))
			print(f"    {ds:<11} AUROC at T={args.image_temperature}: frozen {s0:5.1f}  head {s1:5.1f}")
			np.savez_compressed(
				os.path.join(args.out_root, tag, ds + ".npz"),
				h_cos=c.cpu().numpy(), h_cos_flip=cf.cpu().numpy(),
				img_path=d["img_path"], cls_name=d["cls_name"], label=d["label"].cpu().numpy())
		torch.save(head.state_dict(), os.path.join(args.out_root, tag, "head.pth"))


if __name__ == "__main__":
	main()
