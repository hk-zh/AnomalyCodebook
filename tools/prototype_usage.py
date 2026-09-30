#!/usr/bin/env python3
"""Which prototypes the patch tokens and the global token select, per checkpoint.

tools/global_token_scores.py showed that in the adopted checkpoints every test
image's global token (frozen CLIP global embedding -> image adapter) quantizes
to one and the same prototype. This records, on one dataset,

  patch_counts   [L, K] assignment counts of the patch tokens per layer
  g_idx          [N] prototype chosen by each image's global token
  g_top2         [N, 2] cosine of the global token to its best and second-best prototype

so one can ask whether any patch ever uses the prototype the global token
collapsed onto, i.e. whether L_img_quant can reach the anomaly map at all.
"""
import argparse
import json
import os
import sys

import numpy as np
import torch
import torch.nn.functional as F
import torchvision.transforms as transforms
from torchvision.transforms import InterpolationMode
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import open_clip                                                    # noqa: E402
from dataset import MVTecDataset                                    # noqa: E402
from model import HybridCodebook, LinearLayer                       # noqa: E402
from train import GlobalImageAdaptor                                # noqa: E402
from test import build_dataset, extract_patch_tokens_for_layer, setup_seed  # noqa: E402


def main():
	ap = argparse.ArgumentParser()
	ap.add_argument("--dataset", required=True)
	ap.add_argument("--data_path", required=True)
	ap.add_argument("--checkpoint_path", required=True)
	ap.add_argument("--out", required=True, help="output .npz path")
	ap.add_argument("--split", default="test")
	ap.add_argument("--model", default="ViT-L-14-336")
	ap.add_argument("--pretrained", default="openai")
	ap.add_argument("--config_path", default="./open_clip/model_configs/ViT-L-14-336.json")
	ap.add_argument("--features_list", type=int, nargs="+", default=[6, 12, 18, 24])
	ap.add_argument("--image_size", type=int, default=518)
	ap.add_argument("--test_batch_size", type=int, default=8)
	ap.add_argument("--num_workers", type=int, default=4)
	ap.add_argument("--seed", type=int, default=42)
	args = ap.parse_args()
	setup_seed(args.seed)
	device = "cuda"

	model, _, preprocess = open_clip.create_model_and_transforms(
		args.model, args.image_size, pretrained=args.pretrained)
	model = model.to(device).eval()
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
	if args.dataset == "mvtec":		# the source data; test.py's build_dataset has no mvtec branch
		test_data = MVTecDataset(root=args.data_path, transform=preprocess, target_transform=target_transform_b,
								 target_transform_type=target_transform_type, aug_rate=0.0)
	else:
		test_data = build_dataset(args, preprocess, target_transform_b, target_transform_type)
	loader = torch.utils.data.DataLoader(
		test_data, batch_size=args.test_batch_size, shuffle=False,
		num_workers=args.num_workers, pin_memory=True)

	ckpt = torch.load(args.checkpoint_path, map_location=device)
	saved = ckpt["hybrid_codebook"]["learnable_entries"]
	K = saved.shape[0]
	codebook = HybridCodebook(semantic_embeddings=None, num_learnable=K, embed_dim=saved.shape[1]).to(device)
	with torch.no_grad():
		codebook.learnable_entries.copy_(saved.to(device))
	codebook.eval()
	with open(args.config_path) as f:
		cfg = json.load(f)
	linear = LinearLayer(cfg["vision_cfg"]["width"], cfg["embed_dim"], len(args.features_list), args.model).to(device)
	linear.load_state_dict(ckpt["trainable_linearlayer"])
	linear.eval()
	ia = ckpt["image_adaptor"]
	adaptor = GlobalImageAdaptor(embed_dim=saved.shape[1], hidden_dim=ia["net.1.weight"].shape[0]).to(device)
	adaptor.load_state_dict(ia)
	adaptor.eval()
	protos = F.normalize(saved.to(device).float(), dim=-1)

	patch_counts = torch.zeros(len(args.features_list), K, dtype=torch.long, device=device)
	g_idx, g_top2, labels, cls = [], [], [], []
	for items in tqdm(loader, desc=args.dataset):
		images = items["img"].to(device, non_blocking=True)
		with torch.inference_mode():
			image_features, patch_tokens = model.encode_image(images, args.features_list)
			patch_tokens = linear(patch_tokens)
			for l, pt in enumerate(patch_tokens):
				idx = codebook(extract_patch_tokens_for_layer(pt, args.model).float())["indices"]
				patch_counts[l] += torch.bincount(idx.reshape(-1), minlength=K)
			ga = adaptor(image_features.float())					# normalized
			sim = ga @ protos.t()
			top = sim.topk(2, dim=-1)
		g_idx.extend(top.indices[:, 0].cpu().tolist())
		g_top2.extend(top.values.cpu().numpy())
		labels.extend(int(a) for a in items["anomaly"])
		cls.extend(items["cls_name"])

	os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
	np.savez(args.out, patch_counts=patch_counts.cpu().numpy(), g_idx=np.array(g_idx),
			 g_top2=np.array(g_top2), label=np.array(labels), cls_name=np.array(cls))
	pc = patch_counts.cpu().numpy()
	u, n = np.unique(g_idx, return_counts=True)
	print(f"global token: {len(u)} distinct prototypes; top {dict(zip(u[np.argsort(-n)][:3].tolist(), np.sort(n)[::-1][:3].tolist()))}")
	for j in u[np.argsort(-n)][:3]:
		print(f"  prototype {j}: patch share per layer " + " ".join(f"{100 * pc[l, j] / pc[l].sum():.4f}%" for l in range(len(pc))))
	print(f"live prototypes per layer (patch share > 0): {[(pc[l] > 0).sum() for l in range(len(pc))]}")


if __name__ == "__main__":
	main()
