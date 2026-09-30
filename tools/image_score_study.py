#!/usr/bin/env python3
"""Candidate image-level scores for one checkpoint on one dataset.

The pixel maps match or beat every baseline, but the image score pooled from
them trails on MPDD, MAD-Real and VisA, and the offline read-out study
(tools/readout_study.py) showed that no statistic of the final map closes the
gap on every dataset. Whatever fixes it therefore has to bring in information
the final map throws away. This script runs the exact test.py forward pass once
and records, per image, every label-free candidate we can compute:

  map_topk      top-k mean of the smoothed final map (the current score)
  layer_topk    the same per layer, before the four layers are averaged
  g_cos         frozen CLIP global embedding vs. the text anchors
  cls_cos       each layer's class token through that layer's trained adapter
  clsq_cos      the same, quantized against the codebook first
  meanq_cos     mean-pooled quantized patch features vs. the text anchors
  nov_*         codebook novelty, 1 - cosine to the nearest prototype
  ent_*         normalized assignment entropy U (the temperature signal)

Cosine vectors are stored raw so that any temperature can be applied offline.
Everything is written to one .npz; tools/image_score_fusion.py does the
analysis on CPU. map_topk at rho=0.001, sigma=4 must reproduce the image AUROC
test.py logs for the same checkpoint, which is the check that the forward pass
here is the one we report.
"""
import argparse
import json
import os
import sys
from collections import defaultdict

import numpy as np
import torch
import torch.nn.functional as F
import torchvision.transforms as transforms
from torchvision.transforms import InterpolationMode
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import open_clip                                                    # noqa: E402
from model import LinearLayer, HybridCodebook                       # noqa: E402
from codebooks.utils import get_averaged_text_prompt_embeddings     # noqa: E402
from test import (                                                  # noqa: E402
	build_dataset,
	build_text_prompts,
	get_prompt_dataset_name,
	build_cached_text_prompt_batch,
	extract_patch_tokens_for_layer,
	build_calibration_map,
	entropy_adaptive_softmax,
	gaussian_blur_maps,
	setup_seed,
)

RHOS = (0.0005, 0.001, 0.005, 0.01, 0.05, 0.1)
LAYER_RHOS = (0.001, 0.01)
SIGMAS = (4.0, 8.0)


def topk_mean(flat, rho):
	k = max(1, int(round(rho * flat.shape[1])))
	return torch.topk(flat, k, dim=1).values.mean(dim=1)


def main():
	ap = argparse.ArgumentParser()
	ap.add_argument("--dataset", required=True)
	ap.add_argument("--data_path", required=True)
	ap.add_argument("--checkpoint_path", required=True)
	ap.add_argument("--out", required=True, help="output .npz path")
	ap.add_argument("--split", default="test")
	ap.add_argument("--config_path", default="./open_clip/model_configs/ViT-L-14-336.json")
	ap.add_argument("--model", default="ViT-L-14-336")
	ap.add_argument("--pretrained", default="openai")
	ap.add_argument("--features_list", type=int, nargs="+", default=[6, 12, 18, 24])
	ap.add_argument("--image_size", type=int, default=518)
	ap.add_argument("--temperature", type=float, default=0.01)
	ap.add_argument("--max_temperature", type=float, default=0.1)
	ap.add_argument("--assign_temperature", type=float, default=0.1)
	ap.add_argument("--test_batch_size", type=int, default=8)
	ap.add_argument("--num_workers", type=int, default=4)
	ap.add_argument("--seed", type=int, default=42)
	args = ap.parse_args()
	setup_seed(args.seed)

	device = "cuda" if torch.cuda.is_available() else "cpu"
	use_amp = device == "cuda"

	model, _, preprocess = open_clip.create_model_and_transforms(
		args.model, args.image_size, pretrained=args.pretrained)
	model = model.to(device).eval()
	tokenizer = open_clip.get_tokenizer(args.model)

	target_transform_b = transforms.Compose([
		transforms.Resize((args.image_size, args.image_size)),
		transforms.CenterCrop(args.image_size),
		transforms.ToTensor(),
	])
	target_transform_type = transforms.Compose([
		transforms.Resize((args.image_size, args.image_size), interpolation=InterpolationMode.NEAREST),
		transforms.CenterCrop(args.image_size),
		transforms.PILToTensor(),
		transforms.Lambda(lambda x: x.squeeze(0).long()),
	])
	test_data = build_dataset(args, preprocess, target_transform_b, target_transform_type)
	loader = torch.utils.data.DataLoader(
		test_data, batch_size=args.test_batch_size, shuffle=False,
		num_workers=args.num_workers, pin_memory=(device == "cuda"))
	obj_list = test_data.get_cls_names()

	with torch.inference_mode(), torch.cuda.amp.autocast(enabled=use_amp):
		text_prompts = build_text_prompts(model, obj_list, tokenizer, device, args.dataset)
		semantic_embeddings = get_averaged_text_prompt_embeddings(
			model=model, objs=obj_list, tokenizer=tokenizer, device=device,
			dataset_name=get_prompt_dataset_name(args.dataset), normalize=True)
	text_prompts = {
		k: F.normalize(v.to(device=device, dtype=torch.float32), dim=0).contiguous()
		for k, v in text_prompts.items()
	}
	C_max = max(v.shape[1] for v in text_prompts.values())

	ckpt = torch.load(args.checkpoint_path, map_location=device)
	saved = ckpt["hybrid_codebook"]["learnable_entries"]
	codebook = HybridCodebook(
		semantic_embeddings=None, num_learnable=saved.shape[0],
		embed_dim=semantic_embeddings.shape[-1]).to(device)
	with torch.no_grad():
		codebook.learnable_entries.copy_(saved.to(device))
	codebook.eval()
	protos = F.normalize(codebook.learnable_entries.float(), dim=-1)

	with open(args.config_path) as f:
		cfg = json.load(f)
	linear = LinearLayer(cfg["vision_cfg"]["width"], cfg["embed_dim"],
						 len(args.features_list), args.model).to(device)
	linear.load_state_dict(ckpt["trainable_linearlayer"])
	linear.eval()

	nL = len(args.features_list)
	rec = defaultdict(list)

	def pad_c(x):
		# [B, C] -> [B, C_max], NaN past this product's class count
		if x.shape[-1] == C_max:
			return x
		out = torch.full(x.shape[:-1] + (C_max,), float("nan"), device=x.device)
		out[..., :x.shape[-1]] = x
		return out

	for items in tqdm(loader, desc=f"{args.dataset}"):
		images = items["img"].to(device, non_blocking=True)
		cls_names = list(items["cls_name"])
		B = images.shape[0]

		with torch.inference_mode(), torch.cuda.amp.autocast(enabled=use_amp):
			image_features, patch_tokens = model.encode_image(images, args.features_list)
			cls_raw = [patch_tokens[l][:, 0, :] for l in range(nL)]
			patch_tokens = linear(patch_tokens)
			cls_ad = [F.normalize(linear.fc[l](cls_raw[l]).float(), dim=-1) for l in range(nL)]
			img_g = F.normalize(image_features.float(), dim=-1)

			out = {k: [None] * B for k in (
				"map_topk", "layer_topk", "g_cos", "cls_cos", "clsq_cos",
				"meanq_cos", "nov_max", "nov_top1pct", "nov_mean",
				"ent_max", "ent_top1pct", "ent_mean", "map_mean")}

			groups = defaultdict(list)
			for i, c in enumerate(cls_names):
				groups[c].append(i)

			for cls_name, idx in groups.items():
				it = torch.tensor(idx, device=device)
				tf = build_cached_text_prompt_batch(text_prompts, [cls_name] * len(idx), device)  # [b,D,C]
				b = len(idx)

				g_cos = torch.bmm(img_g.index_select(0, it).unsqueeze(1), tf).squeeze(1)
				cls_cos, clsq_cos, meanq_cos = [], [], []
				nov_max, nov_top, nov_mean = [], [], []
				ent_max, ent_top, ent_mean = [], [], []
				layer_maps = []

				for l in range(nL):
					tok = extract_patch_tokens_for_layer(
						patch_tokens[l].index_select(0, it), args.model)
					ret = codebook(tok, assign_temperature=args.assign_temperature)
					zq = F.normalize(ret["z_q_st"].float(), dim=-1)
					probs = ret["assign_probs"]

					# class token through this layer's adapter, raw and quantized
					c = cls_ad[l].index_select(0, it)
					cls_cos.append(torch.bmm(c.unsqueeze(1), tf).squeeze(1))
					cq = protos[(c @ protos.t()).argmax(dim=-1)]
					clsq_cos.append(torch.bmm(cq.unsqueeze(1), tf).squeeze(1))
					mq = F.normalize(zq.mean(dim=1), dim=-1)
					meanq_cos.append(torch.bmm(mq.unsqueeze(1), tf).squeeze(1))

					L = tok.shape[1]
					k1 = max(1, int(round(0.01 * L)))
					nov = 1.0 - ret["raw_logits"].float().max(dim=-1).values		# [b, L]
					nov_max.append(nov.max(dim=1).values)
					nov_top.append(torch.topk(nov, k1, dim=1).values.mean(dim=1))
					nov_mean.append(nov.mean(dim=1))
					ent = -(probs * torch.log(probs + 1e-8)).sum(-1) / np.log(probs.shape[-1] + 1e-8)
					ent_max.append(ent.max(dim=1).values)
					ent_top.append(torch.topk(ent, k1, dim=1).values.mean(dim=1))
					ent_mean.append(ent.mean(dim=1))

					# pixel map exactly as test.py (score_blend 0, --cali)
					sim = torch.bmm(zq, tf)
					_, _, C = sim.shape
					H = int(np.sqrt(L))
					sim = F.interpolate(sim.permute(0, 2, 1).contiguous().view(b, C, H, H),
										size=(args.image_size, args.image_size),
										mode="bilinear", align_corners=True)
					calib = build_calibration_map(probs, args.image_size)
					seg = entropy_adaptive_softmax(sim, calib, args.temperature, args.max_temperature)
					layer_maps.append(seg[:, 1:].sum(dim=1).float())

				acc = torch.stack(layer_maps, 0).mean(0)
				map_topk = []
				for s in SIGMAS:
					flat = gaussian_blur_maps(acc, s).reshape(b, -1)
					map_topk.append(torch.stack([topk_mean(flat, r) for r in RHOS], 1))
				map_topk = torch.stack(map_topk, 1)									# [b, S, R]
				map_mean = gaussian_blur_maps(acc, SIGMAS[0]).reshape(b, -1).mean(1)
				layer_topk = torch.stack([
					torch.stack([topk_mean(gaussian_blur_maps(m, SIGMAS[0]).reshape(b, -1), r)
								 for r in LAYER_RHOS], 1)
					for m in layer_maps], 1)											# [b, nL, R]

				vals = {
					"map_topk": map_topk,
					"layer_topk": layer_topk,
					"g_cos": pad_c(g_cos),
					"cls_cos": torch.stack([pad_c(x) for x in cls_cos], 1),
					"clsq_cos": torch.stack([pad_c(x) for x in clsq_cos], 1),
					"meanq_cos": torch.stack([pad_c(x) for x in meanq_cos], 1),
					"nov_max": torch.stack(nov_max, 1),
					"nov_top1pct": torch.stack(nov_top, 1),
					"nov_mean": torch.stack(nov_mean, 1),
					"ent_max": torch.stack(ent_max, 1),
					"ent_top1pct": torch.stack(ent_top, 1),
					"ent_mean": torch.stack(ent_mean, 1),
					"map_mean": map_mean,
				}
				for k, v in vals.items():
					v = v.float().cpu().numpy()
					for j, gi in enumerate(idx):
						out[k][gi] = v[j]

		for k, v in out.items():
			rec[k].extend(v)
		rec["cls_name"].extend(cls_names)
		rec["label"].extend(int(a) for a in items["anomaly"])
		rec["img_path"].extend(list(items["img_path"]))
		if "defect_cls" in items:
			rec["defect_cls"].extend(list(items["defect_cls"]))

	os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
	arrays = {k: np.asarray(v) for k, v in rec.items()}
	arrays["rhos"] = np.asarray(RHOS)
	arrays["layer_rhos"] = np.asarray(LAYER_RHOS)
	arrays["sigmas"] = np.asarray(SIGMAS)
	np.savez_compressed(args.out, **arrays)

	# sanity line: must match test.py's logged mean image AUROC for this ckpt
	from sklearn.metrics import roc_auc_score
	s = arrays["map_topk"][:, 0, list(RHOS).index(0.001)]
	aucs = []
	for c in sorted(set(rec["cls_name"])):
		m = arrays["cls_name"] == c
		if len(set(arrays["label"][m])) == 2:
			aucs.append(roc_auc_score(arrays["label"][m], s[m]))
	print(f"[check] {args.dataset}: {len(rec['label'])} images, "
		  f"map_topk@0.001/sigma4 mean image AUROC = {100 * np.mean(aucs):.2f}")


if __name__ == "__main__":
	main()
