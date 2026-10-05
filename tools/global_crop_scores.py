#!/usr/bin/env python3
"""Frozen CLIP global score on image tiles, for small defects.

The adopted image score fuses the map with the frozen CLIP global score of the
whole image at 336 px (tools/global_native_scores.py), where a defect of one ViT
patch at 518 px covers a fraction of a token. Here the same centre-cropped square
is resized to grid * 336 px and cut into grid x grid tiles of 336 px, each scored
at the native resolution like the whole image. Per image this records

  g_crop_cos       [ntiles, C] tile embedding vs. the text anchors
  g_crop_cos_flip  the same for each horizontally flipped tile

in the image order of tools/image_score_study.py, so the files join on img_path.
No trained weight is involved, so one run serves every seed.
"""
import argparse
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
from open_clip.transform import image_transform                    # noqa: E402
from test import build_dataset, build_text_prompts, setup_seed      # noqa: E402


def main():
	ap = argparse.ArgumentParser()
	ap.add_argument("--dataset", required=True)
	ap.add_argument("--data_path", required=True)
	ap.add_argument("--out", required=True, help="output .npz path")
	ap.add_argument("--grid", type=int, default=2, help="tiles per side")
	ap.add_argument("--split", default="test")
	ap.add_argument("--model", default="ViT-L-14-336")
	ap.add_argument("--pretrained", default="openai")
	ap.add_argument("--features_list", type=int, nargs="+", default=[6, 12, 18, 24])
	ap.add_argument("--tile_size", type=int, default=336)
	ap.add_argument("--test_batch_size", type=int, default=8)
	ap.add_argument("--num_workers", type=int, default=4)
	ap.add_argument("--seed", type=int, default=42)
	ap.add_argument("--num_shards", type=int, default=1, help="split the dataset into contiguous shards run as separate jobs")
	ap.add_argument("--shard", type=int, default=0)
	ap.add_argument("--merge", action="store_true", help="concatenate the shard files of --out into --out, in shard order")
	args = ap.parse_args()
	if args.merge:
		merge(args.out, args.num_shards)
		return
	setup_seed(args.seed)
	device = "cuda"

	model, _, _ = open_clip.create_model_and_transforms(args.model, args.tile_size, pretrained=args.pretrained)
	model = model.to(device).eval()
	tokenizer = open_clip.get_tokenizer(args.model)

	size = args.grid * args.tile_size
	args.image_size = size				# build_dataset reads it for the mask transforms
	preprocess = image_transform(size, is_train=False)
	target_transform_b = transforms.Compose([
		transforms.Resize((size, size)), transforms.CenterCrop(size), transforms.ToTensor()])
	target_transform_type = transforms.Compose([
		transforms.Resize((size, size), interpolation=InterpolationMode.NEAREST),
		transforms.CenterCrop(size), transforms.PILToTensor(),
		transforms.Lambda(lambda x: x.squeeze(0).long())])
	test_data = build_dataset(args, preprocess, target_transform_b, target_transform_type)
	obj_list = test_data.get_cls_names()
	if args.num_shards > 1:
		n = len(test_data)
		block = -(-n // args.num_shards)
		test_data = torch.utils.data.Subset(test_data, range(args.shard * block, min(n, (args.shard + 1) * block)))
	loader = torch.utils.data.DataLoader(
		test_data, batch_size=args.test_batch_size, shuffle=False,
		num_workers=args.num_workers, pin_memory=True)

	with torch.inference_mode():
		text_prompts = build_text_prompts(model, obj_list, tokenizer, device, args.dataset)
	text_prompts = {k: F.normalize(v.to(device=device, dtype=torch.float32), dim=0) for k, v in text_prompts.items()}
	C_max = max(v.shape[1] for v in text_prompts.values())

	def pad_c(x):
		if x.shape[-1] == C_max:
			return x
		out = torch.full(x.shape[:-1] + (C_max,), float("nan"), device=x.device)
		out[..., :x.shape[-1]] = x
		return out

	g, t = args.grid, args.tile_size
	rec = defaultdict(list)
	for items in tqdm(loader, desc=f"{args.dataset} grid {g}"):
		images = items["img"].to(device, non_blocking=True)			# [B, 3, g*t, g*t]
		cls_names = list(items["cls_name"])
		B = images.shape[0]
		# [B, 3, g, t, g, t] -> [B*g*g, 3, t, t], tiles in row-major order
		tiles = images.reshape(B, 3, g, t, g, t).permute(0, 2, 4, 1, 3, 5).reshape(B * g * g, 3, t, t)
		with torch.inference_mode():			# fp32: fp16 made the global score batch-size dependent
			vals = {}
			for name, x in (("g_crop_cos", tiles), ("g_crop_cos_flip", torch.flip(tiles, dims=[3]))):
				emb, _ = model.encode_image(x, args.features_list)
				emb = F.normalize(emb.float(), dim=-1).reshape(B, g * g, -1)
				vals[name] = torch.stack([pad_c(emb[i] @ text_prompts[c]) for i, c in enumerate(cls_names)])
		for k, v in vals.items():
			rec[k].extend(v.cpu().numpy())
		rec["cls_name"].extend(cls_names)
		rec["label"].extend(int(a) for a in items["anomaly"])
		rec["img_path"].extend(list(items["img_path"]))

	out = shard_path(args.out, args.shard, args.num_shards) if args.num_shards > 1 else args.out
	os.makedirs(os.path.dirname(os.path.abspath(out)), exist_ok=True)
	np.savez_compressed(out, **{k: np.asarray(v) for k, v in rec.items()})
	print(f"[done] {args.dataset} grid {g}: {len(rec['label'])} images -> {out}")


def shard_path(out, shard, num_shards):
	return out.replace(".npz", f".shard{shard}of{num_shards}.npz")


def merge(out, num_shards):
	parts = [np.load(shard_path(out, i, num_shards)) for i in range(num_shards)]
	np.savez_compressed(out, **{k: np.concatenate([p[k] for p in parts]) for k in parts[0].files})
	print(f"[merged] {num_shards} shards -> {out}: {sum(len(p['label']) for p in parts)} images")


if __name__ == "__main__":
	main()
