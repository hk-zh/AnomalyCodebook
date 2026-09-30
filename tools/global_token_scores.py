#!/usr/bin/env python3
"""Global-token scores for one checkpoint on one dataset.

tools/image_score_fusion.py found that the frozen CLIP global score (g_cos)
complements the map read-out. The trained model already has a codebook-routed
version of that signal which the paper uses only during training: the global
CLIP embedding passed through the residual image adapter and quantized against
the same codebook (train.py, compute_image_level_loss). This records, per image,

  g_cos    frozen CLIP global embedding vs. the text anchors (join check)
  ga_cos   the adapted global embedding vs. the text anchors
  gq_cos   the adapted embedding quantized to its nearest prototype
  gq_idx   that prototype's index

in the same image order as tools/image_score_study.py, so the two .npz files
can be joined on img_path.
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
from model import HybridCodebook                                    # noqa: E402
from train import GlobalImageAdaptor                                # noqa: E402
from test import (                                                  # noqa: E402
	build_dataset,
	build_text_prompts,
	setup_seed,
)


def main():
	ap = argparse.ArgumentParser()
	ap.add_argument("--dataset", required=True)
	ap.add_argument("--data_path", required=True)
	ap.add_argument("--checkpoint_path", required=True)
	ap.add_argument("--out", required=True, help="output .npz path")
	ap.add_argument("--split", default="test")
	ap.add_argument("--model", default="ViT-L-14-336")
	ap.add_argument("--pretrained", default="openai")
	ap.add_argument("--features_list", type=int, nargs="+", default=[6, 12, 18, 24])
	ap.add_argument("--image_size", type=int, default=518)
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
	text_prompts = {
		k: F.normalize(v.to(device=device, dtype=torch.float32), dim=0).contiguous()
		for k, v in text_prompts.items()
	}
	C_max = max(v.shape[1] for v in text_prompts.values())

	ckpt = torch.load(args.checkpoint_path, map_location=device)
	saved = ckpt["hybrid_codebook"]["learnable_entries"]
	codebook = HybridCodebook(
		semantic_embeddings=None, num_learnable=saved.shape[0],
		embed_dim=saved.shape[1]).to(device)
	with torch.no_grad():
		codebook.learnable_entries.copy_(saved.to(device))
	codebook.eval()

	ia = ckpt["image_adaptor"]
	adaptor = GlobalImageAdaptor(
		embed_dim=saved.shape[1], hidden_dim=ia["net.1.weight"].shape[0]).to(device)
	adaptor.load_state_dict(ia)
	adaptor.eval()

	rec = defaultdict(list)

	def pad_c(x):
		if x.shape[-1] == C_max:
			return x
		out = torch.full(x.shape[:-1] + (C_max,), float("nan"), device=x.device)
		out[..., :x.shape[-1]] = x
		return out

	for items in tqdm(loader, desc=f"{args.dataset}"):
		images = items["img"].to(device, non_blocking=True)
		cls_names = list(items["cls_name"])

		with torch.inference_mode(), torch.cuda.amp.autocast(enabled=use_amp):
			image_features, _ = model.encode_image(images, args.features_list)
			g = image_features.float()
			ga = adaptor(g)										# normalized
			ret = codebook(ga.unsqueeze(1))
			gq = F.normalize(ret["z_q_st"].squeeze(1).float(), dim=-1)
			gq_idx = ret["indices"].reshape(-1)
			g = F.normalize(g, dim=-1)

			vals = {}
			for name, v in (("g_cos", g), ("ga_cos", ga), ("gq_cos", gq)):
				cos = []
				for i, c in enumerate(cls_names):
					cos.append(pad_c(v[i:i + 1] @ text_prompts[c]))
				vals[name] = torch.cat(cos, 0)

		for k, v in vals.items():
			rec[k].extend(v.float().cpu().numpy())
		rec["gq_idx"].extend(gq_idx.cpu().numpy().tolist())
		rec["cls_name"].extend(cls_names)
		rec["label"].extend(int(a) for a in items["anomaly"])
		rec["img_path"].extend(list(items["img_path"]))

	os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
	arrays = {k: np.asarray(v) for k, v in rec.items()}
	np.savez_compressed(args.out, **arrays)
	print(f"[done] {args.dataset}: {len(rec['label'])} images, "
		  f"{len(set(rec['gq_idx']))} distinct global prototypes")


if __name__ == "__main__":
	main()
