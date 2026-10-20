#!/usr/bin/env python3
"""Baseline anomaly maps for the images of the qualitative figure.

For each image in results/qualitative/s42.npz (drawn by tools/make_qualitative.py)
this runs one baseline from its own code base and MVTec-AD checkpoint, the same
ones as in the transfer tables, on the same 518 x 518 input tensor our model sees:

  anomalyclip  authors' MVTec-AD-trained weights, test settings of their test.sh
               (layer 24, DPAM at 20). Their train.sh writes the MVTec-AD run to
               9_12_4_multiscale_visa and the VisA run to 9_12_4_multiscale; until
               2026-10-10 this tool loaded the latter (maps kept as *_visaw.npz)
  dapo         checkpoint/mvtec/epoch_4; prompt states from DAPO's per-dataset
               config, or the target's defect names where it has none (MAD,
               Real-IAD)
  multiads     our one-epoch MVTec-AD training with its own recipe (exps/mvtec/
               epoch_1), unfiltered prompts (MultiADS, not MultiADS-F)

Every map gets the same Gaussian smoothing (sigma=4) as ours. Each method runs in
its own process with its repository first on sys.path, so their modules (each has
its own open_clip, model, test) cannot collide.

Usage:
    python tools/qual_baselines.py --method anomalyclip
"""
import argparse
import os
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
BASE = os.path.join(ROOT, "baselines")
MEAN = np.array([0.48145466, 0.4578275, 0.40821073])
STD = np.array([0.26862954, 0.26130258, 0.27577711])
DS = {"VisA": "visa", "MPDD": "mpdd", "MAD-Sim": "mad_sim", "MAD-Real": "mad_real", "Real-IAD": "real_iad",
	  "Brain MRI": "bmad_brain", "Liver CT": "bmad_liver", "Retinal OCT": "bmad_resc", "MVTec-LOCO": "loco"}


def inputs(path):
	import torch
	d = np.load(path, allow_pickle=False)
	x = (d["image"].astype(np.float32) / 255.0 - MEAN) / STD			# N, H, W, 3
	return torch.from_numpy(x.transpose(0, 3, 1, 2)).float(), list(d["label"]), list(d["cls_name"])


def anomalyclip(x, labels, cls_names, device):
	import torch
	sys.path.insert(0, os.path.join(BASE, "AnomalyCLIP"))
	os.chdir(os.path.join(BASE, "AnomalyCLIP"))
	import AnomalyCLIP_lib
	from prompt_ensemble import AnomalyCLIP_PromptLearner
	params = {"Prompt_length": 12, "learnabel_text_embedding_depth": 9, "learnabel_text_embedding_length": 4}
	model, _ = AnomalyCLIP_lib.load("ViT-L/14@336px", device=device, design_details=params)
	model.eval()
	learner = AnomalyCLIP_PromptLearner(model.to("cpu"), params)
	learner.load_state_dict(torch.load("checkpoints/9_12_4_multiscale_visa/epoch_15.pth")["prompt_learner"])
	learner.to(device)
	model.to(device)
	model.visual.DAPM_replace(DPAM_layer=20)
	with torch.no_grad():
		prompts, tokenized, compound = learner(cls_id=None)
		text = model.encode_text_learn(prompts, tokenized, compound).float()
		text = torch.stack(torch.chunk(text, dim=0, chunks=2), dim=1)
		text = text / text.norm(dim=-1, keepdim=True)
		out = []
		for i in range(len(x)):
			_, patches = model.encode_image(x[i:i + 1].to(device), [24], DPAM_layer=20)
			maps = []
			for p in patches:
				p = p / p.norm(dim=-1, keepdim=True)
				sim, _ = AnomalyCLIP_lib.compute_similarity(p, text[0])
				sm = AnomalyCLIP_lib.get_similarity_map(sim[:, 1:, :], x.shape[-1])
				maps.append((sm[..., 1] + 1 - sm[..., 0]) / 2.0)
			out.append(torch.stack(maps).sum(0)[0].float().cpu().numpy())
	return out


def dapo(x, labels, cls_names, device):
	import torch
	sys.path.insert(0, os.path.join(BASE, "DAPO"))
	os.chdir(os.path.join(BASE, "DAPO"))
	import test as dapo_test
	extra = {"mad_sim": ["burrs", "missing", "stains"], "mad_real": ["burrs", "missing", "stains"],
			 "real_iad": ["pit", "scratch", "deformation", "abrasion", "damage", "missing", "foreign", "contamination"]}
	key = {"loco": "mvtec_loco"}
	out = [None] * len(labels)
	for lab in dict.fromkeys(labels):		# build DAPO's prompt components once per dataset
		ds = key.get(DS[lab], DS[lab])
		args = dapo_test.create_test_args_parser().parse_args(
			["--dataset", ds if ds not in extra else "mvtec", "--checkpoint_path", "./checkpoint/mvtec/", "--epoch", "4",
			 "--image_size", str(x.shape[-1]), "--model_name", "ViT-L-14-336-quickgelu", "--pretrained", "openai",
			 "--mode", "zero_shot"])
		tester = dapo_test.Tester(args)
		if ds in extra:
			tester.dataset_configs[ds] = {"prompt_states": extra[ds], "dataset_class": "Dataset"}
			tester.args.dataset = ds
		tester.model_components = tester.setup_model()
		tester.learnable_components = tester.setup_learnable_components()
		tester.load_checkpoint("./checkpoint/mvtec/epoch_4.pth")
		for i in [i for i, l in enumerate(labels) if l == lab]:
			with torch.no_grad():
				amap, _ = tester.forward_pass(x[i:i + 1].to(device))
			out[i] = amap[0].float().cpu().numpy()
		del tester
		torch.cuda.empty_cache()
	return out


def multiads(x, labels, cls_names, device):
	import importlib
	import json
	import torch
	import torch.nn.functional as F
	sys.path.insert(0, os.path.join(BASE, "MultiADS"))
	os.chdir(os.path.join(BASE, "MultiADS"))
	import open_clip
	from model import LinearLayer
	mods = {"visa": "prompts.prompt_ensemble_visa_19cls_test", "mpdd": "prompts.new_prompt_ensemble_mpdd",
			"mad_sim": "prompts.prompt_ensemble_mad_sim", "mad_real": "prompts.prompt_ensemble_mad_real",
			"real_iad": "prompts.prompt_ensemble_real_IAD_simple", "bmad_brain": "prompts.prompt_ensemble_bmad",
			"bmad_liver": "prompts.prompt_ensemble_bmad", "bmad_resc": "prompts.prompt_ensemble_bmad",
			"loco": "prompts.prompt_ensemble_loco_object_agnostic"}
	size = x.shape[-1]
	model, _, _ = open_clip.create_model_and_transforms("ViT-L-14-336", size, pretrained="openai")
	model = model.to(device).eval()
	tokenizer = open_clip.get_tokenizer("ViT-L-14-336")
	cfg = json.load(open("./open_clip/model_configs/ViT-L-14-336.json"))
	linear = LinearLayer(cfg["vision_cfg"]["width"], cfg["embed_dim"], 4, "ViT-L-14-336").to(device)
	linear.load_state_dict(torch.load("./exps/mvtec/epoch_1.pth", map_location=device)["trainable_linearlayer"])
	linear.eval()
	out, cache = [], {}
	for i, (lab, cls) in enumerate(zip(labels, cls_names)):
		with torch.no_grad(), torch.amp.autocast("cuda"):
			if (lab, cls) not in cache:
				enc = importlib.import_module(mods[DS[lab]]).encode_text_with_prompt_ensemble
				cache[(lab, cls)] = enc(model, [cls], tokenizer, device)[cls].unsqueeze(0)		# 1, D, C
			text = cache[(lab, cls)]
			_, patches = model.encode_image(x[i:i + 1].to(device), [6, 12, 18, 24])
			patches = linear(patches)
			maps = []
			for p in patches:
				p = p / p.norm(dim=-1, keepdim=True)
				logit = (p @ text) * 100.0
				B, L, C = logit.shape
				H = int(np.sqrt(L))
				logit = F.interpolate(logit.permute(0, 2, 1).reshape(B, C, H, H), size=size, mode="bilinear", align_corners=True)
				maps.append(torch.softmax(logit.float(), dim=1)[:, 1:].sum(1))
		out.append(torch.stack(maps).sum(0)[0].float().cpu().numpy())
	return out


def main():
	ap = argparse.ArgumentParser()
	ap.add_argument("--method", required=True, choices=["anomalyclip", "dapo", "multiads"])
	ap.add_argument("--inp", default=os.path.join(ROOT, "results/qualitative/s42.npz"))
	ap.add_argument("--out_dir", default=None, help="default: the directory of --inp")
	args = ap.parse_args()
	args.inp = os.path.abspath(args.inp)		# each method chdirs into its own repository
	args.out_dir = os.path.abspath(args.out_dir) if args.out_dir else os.path.dirname(args.inp)
	from scipy.ndimage import gaussian_filter
	x, labels, cls_names = inputs(args.inp)
	maps = {"anomalyclip": anomalyclip, "dapo": dapo, "multiads": multiads}[args.method](x, labels, cls_names, "cuda")
	maps = np.stack([gaussian_filter(m, sigma=4) for m in maps]).astype(np.float16)
	out = os.path.join(args.out_dir, f"{args.method}.npz")
	np.savez_compressed(out, amap=maps, label=np.array(labels))
	print(f"[saved] {out}")


if __name__ == "__main__":
	main()
