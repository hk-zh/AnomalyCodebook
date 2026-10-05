#!/usr/bin/env python3
"""Frozen CLIP global score at a chosen input resolution.

tools/image_score_fusion.py found that the frozen CLIP global score (g_cos)
complements the map read-out, but tools/image_score_study.py records it at the
518 px used for the patch maps, where ViT-L-14-336 runs on interpolated position
embeddings. The global embedding needs no patch grid, so it can be taken at the
native 336 px instead. No trained weight is involved, so one run serves every
seed. Per image this records

  g_cos       global embedding vs. the text anchors
  g_cos_flip  the same for the horizontally flipped image

in the same image order as tools/image_score_study.py, so the two .npz files
can be joined on img_path. With --save_emb it also keeps the raw global
embeddings (emb, emb_flip) and the text anchors (text_cls, text_emb), which
tools/train_global_head.py trains and evaluates on. --dataset mvtec reads the
MVTec split that train.py trains on (mosaic augmentation off).
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
from dataset import MVTecDataset                                    # noqa: E402
from test import (                                                  # noqa: E402
	build_dataset,
	build_text_prompts,
	setup_seed,
)


def main():
	ap = argparse.ArgumentParser()
	ap.add_argument("--dataset", required=True)
	ap.add_argument("--data_path", required=True)
	ap.add_argument("--out", required=True, help="output .npz path")
	ap.add_argument("--split", default="test")
	ap.add_argument("--model", default="ViT-L-14-336")
	ap.add_argument("--pretrained", default="openai")
	ap.add_argument("--features_list", type=int, nargs="+", default=[6, 12, 18, 24])
	ap.add_argument("--image_size", type=int, default=336)
	ap.add_argument("--test_batch_size", type=int, default=16)
	ap.add_argument("--num_workers", type=int, default=4)
	ap.add_argument("--seed", type=int, default=42)
	ap.add_argument("--save_emb", action="store_true", help="also save raw embeddings and text anchors")
	ap.add_argument("--fp32", action="store_true",
					help="image encoder in fp32 (fp16 autocast makes the embeddings depend on the batch size)")
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
	if args.dataset == "mvtec":		# test.build_dataset passes a kwarg MVTecDataset lacks
		test_data = MVTecDataset(
			root=args.data_path, transform=preprocess, target_transform=target_transform_b,
			target_transform_type=target_transform_type, aug_rate=-1, mode="test")
	else:
		test_data = build_dataset(args, preprocess, target_transform_b, target_transform_type)
	loader = torch.utils.data.DataLoader(
		test_data, batch_size=args.test_batch_size, shuffle=False,
		num_workers=args.num_workers, pin_memory=(device == "cuda"))
	obj_list = test_data.get_cls_names()

	# with --fp32 the text anchors are fp32 too, as in tools/global_crop_scores.py and
	# test.py's global pass, so whole-image and tile scores share one set of anchors
	with torch.inference_mode(), torch.cuda.amp.autocast(enabled=use_amp and not args.fp32):
		text_prompts = build_text_prompts(model, obj_list, tokenizer, device, args.dataset)
	text_prompts = {
		k: F.normalize(v.to(device=device, dtype=torch.float32), dim=0).contiguous()
		for k, v in text_prompts.items()
	}
	C_max = max(v.shape[1] for v in text_prompts.values())

	rec = defaultdict(list)

	def pad_c(x):
		if x.shape[-1] == C_max:
			return x
		out = torch.full(x.shape[:-1] + (C_max,), float("nan"), device=x.device)
		out[..., :x.shape[-1]] = x
		return out

	for items in tqdm(loader, desc=f"{args.dataset}@{args.image_size}"):
		images = items["img"].to(device, non_blocking=True)
		cls_names = list(items["cls_name"])

		with torch.inference_mode(), torch.cuda.amp.autocast(enabled=use_amp and not args.fp32):
			vals = {}
			for name, x in (("g_cos", images), ("g_cos_flip", torch.flip(images, dims=[3]))):
				g, _ = model.encode_image(x, args.features_list)
				if args.save_emb:
					vals[name.replace("g_cos", "emb")] = g.half()
				g = F.normalize(g.float(), dim=-1)
				vals[name] = torch.cat(
					[pad_c(g[i:i + 1] @ text_prompts[c]) for i, c in enumerate(cls_names)], 0)

		for k, v in vals.items():
			rec[k].extend(v.cpu().numpy() if k.startswith("emb") else v.float().cpu().numpy())
		rec["cls_name"].extend(cls_names)
		rec["label"].extend(int(a) for a in items["anomaly"])
		rec["img_path"].extend(list(items["img_path"]))

	os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
	arrays = {k: np.asarray(v) for k, v in rec.items()}
	if args.save_emb:
		arrays["text_cls"] = np.asarray(sorted(text_prompts))
		arrays["text_emb"] = np.stack([
			pad_c(text_prompts[c].unsqueeze(0)).squeeze(0).cpu().numpy() for c in arrays["text_cls"]])
	np.savez_compressed(args.out, **arrays)
	print(f"[done] {args.dataset}@{args.image_size}: {len(rec['label'])} images")


if __name__ == "__main__":
	main()
