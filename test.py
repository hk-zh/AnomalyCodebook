# Copyright (c) 2025 Robert Bosch GmbH
# SPDX-License-Identifier: AGPL-3.0

import os
import cv2
import json
import torch
import random
import logging
import argparse
import numpy as np
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor, as_completed

from skimage import measure
from tabulate import tabulate
import torch.nn.functional as F
import torchvision.transforms as transforms
from torchvision.transforms import InterpolationMode

from sklearn.metrics import (
	auc,
	roc_auc_score,
	average_precision_score,
	precision_recall_curve,
)

import open_clip
from open_clip.transform import image_transform

from model import LinearLayer, HybridCodebook
from dataset import VisaDatasetTest, MVTecDataset, MPDDDataset, MADDataset, RealIADDataset_v2, BMADDataset, MVTecLOCODataset, GoodsADDataset
from codebooks.utils import get_averaged_text_prompt_embeddings

from prompts.prompt_ensemble_mvtec_object_agnostic import encode_text_with_prompt_ensemble as encode_text_with_prompt_ensemble_mvtec
from prompts.prompt_ensemble_visa_object_agnostic import encode_text_with_prompt_ensemble as encode_text_with_prompt_ensemble_visa
from prompts.prompt_ensemble_mpdd_object_agnostic import encode_text_with_prompt_ensemble as encode_text_with_prompt_ensemble_mpdd
from prompts.prompt_ensemble_real_IAD_simple import encode_text_with_prompt_ensemble as encode_text_with_prompt_ensemble_real_iad
from prompts.prompt_ensemble_mad_sim import encode_text_with_prompt_ensemble as encode_text_with_prompt_ensemble_mad_sim
from prompts.prompt_ensemble_mad_real import encode_text_with_prompt_ensemble as encode_text_with_prompt_ensemble_mad_real
from prompts.prompt_ensemble_bmad import encode_text_with_prompt_ensemble as encode_text_with_prompt_ensemble_bmad
from prompts.prompt_ensemble_loco_object_agnostic import encode_text_with_prompt_ensemble as encode_text_with_prompt_ensemble_loco
from prompts.prompt_ensemble_goodsad_object_agnostic import encode_text_with_prompt_ensemble as encode_text_with_prompt_ensemble_goodsad

from tqdm import tqdm


class GlobalImageAdaptor(torch.nn.Module):
	def __init__(self, embed_dim, hidden_dim=None, dropout=0.1):
		super().__init__()

		if hidden_dim is None:
			hidden_dim = embed_dim

		self.net = torch.nn.Sequential(
			torch.nn.LayerNorm(embed_dim),
			torch.nn.Linear(embed_dim, hidden_dim),
			torch.nn.GELU(),
			torch.nn.Dropout(dropout),
			torch.nn.Linear(hidden_dim, embed_dim),
		)

	def forward(self, x):
		x = x.float()
		return F.normalize(x + self.net(x), dim=-1)


def setup_seed(seed: int) -> None:
	torch.manual_seed(seed)
	torch.cuda.manual_seed_all(seed)
	np.random.seed(seed)
	random.seed(seed)
	torch.backends.cudnn.deterministic = True
	torch.backends.cudnn.benchmark = False


def setup_speed(seed: int) -> None:
	torch.manual_seed(seed)
	torch.cuda.manual_seed_all(seed)
	np.random.seed(seed)
	random.seed(seed)

	torch.backends.cudnn.deterministic = False
	torch.backends.cudnn.benchmark = True

	if torch.cuda.is_available():
		torch.backends.cuda.matmul.allow_tf32 = True
		torch.backends.cudnn.allow_tf32 = True


def normalize(pred: np.ndarray, max_value=None, min_value=None) -> np.ndarray:
	if max_value is None or min_value is None:
		den = pred.max() - pred.min()
		if den < 1e-12:
			return np.zeros_like(pred)
		return (pred - pred.min()) / den

	den = max_value - min_value
	if den < 1e-12:
		return np.zeros_like(pred)
	return (pred - min_value) / den


def apply_ad_scoremap(image: np.ndarray, scoremap: np.ndarray, alpha: float = 0.5) -> np.ndarray:
	np_image = np.asarray(image, dtype=float)
	scoremap = (scoremap * 255).astype(np.uint8)
	scoremap = cv2.applyColorMap(scoremap, cv2.COLORMAP_JET)
	scoremap = cv2.cvtColor(scoremap, cv2.COLOR_BGR2RGB)
	return (alpha * np_image + (1 - alpha) * scoremap).astype(np.uint8)


def cal_pro_score(masks: np.ndarray, amaps: np.ndarray, max_step: int = 200, expect_fpr: float = 0.3) -> float:
	binary_amaps = np.zeros_like(amaps, dtype=bool)
	min_th, max_th = amaps.min(), amaps.max()

	if abs(max_th - min_th) < 1e-12:
		return 0.0

	delta = (max_th - min_th) / max_step
	pros, fprs = [], []

	for th in np.arange(min_th, max_th, delta):
		binary_amaps[amaps <= th] = 0
		binary_amaps[amaps > th] = 1

		pro = []
		for binary_amap, mask in zip(binary_amaps, masks):
			for region in measure.regionprops(measure.label(mask)):
				tp_pixels = binary_amap[region.coords[:, 0], region.coords[:, 1]].sum()
				pro.append(tp_pixels / region.area)

		inverse_masks = 1 - masks
		fp_pixels = np.logical_and(inverse_masks, binary_amaps).sum()
		fpr = fp_pixels / max(inverse_masks.sum(), 1)

		pros.append(np.mean(pro) if len(pro) > 0 else 0.0)
		fprs.append(fpr)

	pros = np.array(pros)
	fprs = np.array(fprs)

	valid = fprs < expect_fpr
	if valid.sum() < 2:
		return 0.0

	fprs = fprs[valid]
	pros = pros[valid]

	den = fprs.max() - fprs.min()
	if den < 1e-12:
		return 0.0

	fprs = (fprs - fprs.min()) / den
	return auc(fprs, pros)


def gaussian_blur_maps(maps: torch.Tensor, sigma: float) -> torch.Tensor:
	"""Separable Gaussian blur on a batch of anomaly maps [B, H, W].

	Without this the image score is an extreme-value statistic: at rho=0.001 the
	pooled set is about 1.4 tokens wide, so a single spurious hot patch decides
	the image. Smoothing makes the read-out reflect a region instead, which is
	also what the April-GAN evaluation lineage does.
	"""
	if sigma is None or sigma <= 0:
		return maps
	radius = max(1, int(round(3.0 * sigma)))
	x = torch.arange(-radius, radius + 1, device=maps.device, dtype=maps.dtype)
	kernel = torch.exp(-(x ** 2) / (2.0 * sigma * sigma))
	kernel = kernel / kernel.sum()
	m = maps.unsqueeze(1)
	m = F.conv2d(m, kernel.view(1, 1, 1, -1), padding=(0, radius))
	m = F.conv2d(m, kernel.view(1, 1, -1, 1), padding=(radius, 0))
	return m.squeeze(1)


def build_logger(save_path: str) -> logging.Logger:
	os.makedirs(save_path, exist_ok=True)
	log_path = os.path.join(save_path, "log.txt")

	root_logger = logging.getLogger()
	for handler in root_logger.handlers[:]:
		root_logger.removeHandler(handler)
	root_logger.setLevel(logging.WARNING)

	logger = logging.getLogger("test")
	logger.handlers.clear()
	logger.setLevel(logging.INFO)
	logger.propagate = False

	formatter = logging.Formatter(
		"%(asctime)s.%(msecs)03d - %(levelname)s: %(message)s",
		datefmt="%y-%m-%d %H:%M:%S"
	)

	file_handler = logging.FileHandler(log_path, mode="a")
	file_handler.setFormatter(formatter)
	logger.addHandler(file_handler)

	console_handler = logging.StreamHandler()
	console_handler.setFormatter(formatter)
	logger.addHandler(console_handler)

	return logger


def build_dataset(args, preprocess, target_transform_b, target_transform_type):
	if args.dataset == "mvtec":
		return MVTecDataset(
			root=args.data_path,
			transform=preprocess,
			target_transform_b=target_transform_b,
			target_transform_type=target_transform_type,
			aug_rate=-1,
			mode="test"
		)

	if args.dataset == "visa":
		return VisaDatasetTest(
			root=args.data_path,
			transform=preprocess,
			target_transform=target_transform_b,
			mode="test"
		)

	if args.dataset == "mpdd":
		return MPDDDataset(
			root=args.data_path,
			transform=preprocess,
			target_transform=target_transform_b,
			target_transform_type=target_transform_type,
			mode="test"
		)

	if args.dataset == "mad_sim":
		return MADDataset(
			root=args.data_path,
			transform=preprocess,
			target_transform=target_transform_b,
			target_transform_type=target_transform_type,
			mode="test",
			datatype="sim"
		)

	if args.dataset == "mad_real":
		return MADDataset(
			root=args.data_path,
			transform=preprocess,
			target_transform=target_transform_b,
			target_transform_type=target_transform_type,
			mode="test",
			datatype="real"
		)

	if args.dataset == "bmad":
		return BMADDataset(
			root=args.data_path,
			transform=preprocess,
			target_transform=target_transform_b,
			target_transform_type=target_transform_type,
			mode=args.split
		)

	if args.dataset == "loco":
		return MVTecLOCODataset(
			root=args.data_path,
			transform=preprocess,
			target_transform=target_transform_b,
			target_transform_type=target_transform_type,
			mode=args.split
		)

	if args.dataset == "goodsad":
		return GoodsADDataset(
			root=args.data_path,
			transform=preprocess,
			target_transform=target_transform_b,
			target_transform_type=target_transform_type,
			mode=args.split
		)

	if args.dataset == "real_iad":
		return RealIADDataset_v2(
			root=args.data_path,
			transform=preprocess,
			target_transform=target_transform_b,
			target_transform_type=target_transform_type,
			mode="test"
		)

	raise ValueError(f"Unsupported dataset: {args.dataset}")


def get_prompt_dataset_name(dataset_name: str) -> str:
	if dataset_name == "mvtec":
		return "mvtec_ad"
	if dataset_name == "visa":
		return "visa"
	if dataset_name == "mpdd":
		return "mpdd"
	if dataset_name == "real_iad":
		return "real_iad"

	if dataset_name == "mad_sim":
		return "mad_sim"

	if dataset_name == "mad_real":
		return "mad_real"

	if dataset_name == "bmad":
		return "bmad"

	if dataset_name == "loco":
		return "loco"

	if dataset_name == "goodsad":
		return "goodsad"

	raise ValueError(f"Unsupported dataset for prompt embeddings: {dataset_name}")


def build_text_prompts(model, obj_list, tokenizer, device, dataset_name):
	with torch.no_grad():
		if dataset_name == "mvtec":
			return encode_text_with_prompt_ensemble_mvtec(model, obj_list, tokenizer, device)

		if dataset_name == "visa":
			return encode_text_with_prompt_ensemble_visa(model, obj_list, tokenizer, device)

		if dataset_name == "mpdd":
			return encode_text_with_prompt_ensemble_mpdd(model, obj_list, tokenizer, device)

		if dataset_name == "real_iad":
			return encode_text_with_prompt_ensemble_real_iad(model, obj_list, tokenizer, device)

		if dataset_name == "mad_sim":
			return encode_text_with_prompt_ensemble_mad_sim(model, obj_list, tokenizer, device)

		if dataset_name == "mad_real":
			return encode_text_with_prompt_ensemble_mad_real(model, obj_list, tokenizer, device)

		if dataset_name == "bmad":
			return encode_text_with_prompt_ensemble_bmad(model, obj_list, tokenizer, device)

		if dataset_name == "loco":
			return encode_text_with_prompt_ensemble_loco(model, obj_list, tokenizer, device)

		if dataset_name == "goodsad":
			return encode_text_with_prompt_ensemble_goodsad(model, obj_list, tokenizer, device)

	raise ValueError(f"Unsupported dataset for text prompts: {dataset_name}")


def extract_patch_tokens_for_layer(patch_token, model_name: str):
	if patch_token.ndim == 3:
		if "ViT" in model_name and patch_token.shape[1] > 1:
			spatial_len = patch_token.shape[1] - 1
			side = int(np.sqrt(spatial_len))
			if side * side == spatial_len:
				return patch_token[:, 1:, :]
		return patch_token

	if patch_token.ndim == 4:
		B, C, H, W = patch_token.shape
		return patch_token.view(B, C, H * W).permute(0, 2, 1).contiguous()

	raise ValueError(f"Unsupported patch token shape: {patch_token.shape}")


def build_calibration_map(assign_probs, img_size):
	"""
	assign_probs: [B, L, K]

	return:
		calib_map: [B, H_img, W_img]

	Entropy-only uncertainty map:
		0 = confident codebook assignment
		1 = uncertain codebook assignment
	"""
	B, L, K = assign_probs.shape
	H = int(np.sqrt(L))

	if H * H != L:
		raise ValueError(f"L={L} is not square.")

	entropy = -(assign_probs * torch.log(assign_probs + 1e-8)).sum(dim=-1)
	entropy = entropy / np.log(K + 1e-8)

	calib_map = entropy.view(B, 1, H, H)
	calib_map = F.interpolate(
		calib_map,
		size=(img_size, img_size),
		mode="bilinear",
		align_corners=False
	).squeeze(1)

	return calib_map


def entropy_adaptive_softmax(sim_logits, calib_map, min_temperature=0.01, max_temperature=0.1):
	"""
	sim_logits: [B, C, H, W], raw similarity logits, NOT divided by temperature
	calib_map: [B, H, W], normalized entropy uncertainty in [0, 1]

	return:
		seg_prob: [B, C, H, W]

	Local temperature:
		T(x) = T_min + U(x) * (T_max - T_min)

	U(x)=0 -> T_min
	U(x)=1 -> T_max
	"""
	temp_map = min_temperature + (max_temperature - min_temperature) * calib_map.unsqueeze(1)
	seg_prob = torch.softmax(sim_logits / temp_map, dim=1)
	return seg_prob


def anomaly_map_from_tokens(patch_tokens, idx_tensor, text_features_g, codebooks, args):
	"""Layer-averaged anomaly map of one product group before smoothing, computed
	as in the main loop of test() but without its diagnostics. Used for the
	mirrored pass of --tta_flip."""
	acc_map = None
	for layer_idx in range(len(patch_tokens)):
		layer_tokens = extract_patch_tokens_for_layer(
			patch_tokens[layer_idx].index_select(0, idx_tensor), args.model)
		ret = codebooks[layer_idx](
			layer_tokens,
			assign_temperature=args.assign_temperature,
			semantic_bias=args.semantic_bias,
			assign_mode=args.assign_mode
		)
		patch_used = F.normalize(ret["z_q_st"], dim=-1)
		if args.score_blend > 0.0:
			raw_patch = F.normalize(layer_tokens, dim=-1)
			patch_used = F.normalize(
				args.score_blend * raw_patch + (1.0 - args.score_blend) * patch_used, dim=-1)
		sim_logits = torch.bmm(patch_used, text_features_g)
		Bg, L, C = sim_logits.shape
		H = int(np.sqrt(L))
		sim_logits = F.interpolate(
			sim_logits.permute(0, 2, 1).contiguous().view(Bg, C, H, H),
			size=(args.image_size, args.image_size),
			mode="bilinear",
			align_corners=True
		)
		if args.cali:
			calib_map = build_calibration_map(assign_probs=ret["assign_probs"], img_size=args.image_size)
			seg_prob = entropy_adaptive_softmax(
				sim_logits=sim_logits,
				calib_map=calib_map,
				min_temperature=args.temperature,
				max_temperature=args.max_temperature
			)
		else:
			seg_prob = torch.softmax(sim_logits / args.temperature, dim=1)
		layer_map = seg_prob[:, 1:, :, :].sum(dim=1)
		acc_map = layer_map if acc_map is None else acc_map + layer_map
	return acc_map / len(patch_tokens)


def compute_global_image_score(
	image_features,
	text_features,
	image_adaptor,
	hybrid_codebook,
	temperature,
	assign_temperature,
):
	"""
	image_features: [B, D], frozen CLIP global image feature
	text_features: [B, D, C], class 0 normal/good, classes 1..C-1 anomaly

	return:
		image_anomaly_score: [B]
	"""
	image_features = image_features.float()
	text_features = text_features.float()

	adapted_image_features = image_adaptor(image_features)	# [B, D]

	global_tokens = adapted_image_features.unsqueeze(1)		# [B, 1, D]

	ret = hybrid_codebook(
		global_tokens,
		assign_temperature=assign_temperature
	)

	global_q = ret["z_q_st"].squeeze(1)						# [B, D]
	global_q = F.normalize(global_q, dim=-1)

	image_logits = torch.bmm(
		global_q.unsqueeze(1),
		text_features
	).squeeze(1) / temperature								# [B, C]

	normal_logit = image_logits[:, 0]
	anomaly_logit = torch.logsumexp(image_logits[:, 1:], dim=1)

	binary_anomaly_logit = anomaly_logit - normal_logit
	image_anomaly_score = torch.sigmoid(binary_anomaly_logit)

	return image_anomaly_score


def save_visualization(image_path: str, anomaly_map: np.ndarray, img_size: int, save_path: str, cls_name: str) -> None:
	cls = image_path.split("/")[-2]
	filename = image_path.split("/")[-1]

	vis = cv2.imread(image_path)
	vis = cv2.resize(vis, (img_size, img_size))
	vis = cv2.cvtColor(vis, cv2.COLOR_BGR2RGB)

	mask = normalize(anomaly_map[0])
	vis = apply_ad_scoremap(vis, mask)
	vis = cv2.cvtColor(vis, cv2.COLOR_RGB2BGR)

	save_vis_dir = os.path.join(save_path, "imgs", cls_name, cls)
	os.makedirs(save_vis_dir, exist_ok=True)
	cv2.imwrite(os.path.join(save_vis_dir, filename), vis)


def evaluate_one_object_metric(payload):
	obj, gt_px, pr_px, gt_sp, pr_sp = payload

	gt_px = np.array(gt_px)
	pr_px = np.array(pr_px)
	gt_sp = np.array(gt_sp)
	pr_sp = np.array(pr_sp)

	auroc_px = roc_auc_score(gt_px.ravel(), pr_px.ravel())
	auroc_sp = roc_auc_score(gt_sp, pr_sp)
	ap_sp = average_precision_score(gt_sp, pr_sp)
	ap_px = average_precision_score(gt_px.ravel(), pr_px.ravel())

	precisions, recalls, _ = precision_recall_curve(gt_sp, pr_sp)
	f1_scores = (2 * precisions * recalls) / (precisions + recalls + 1e-12)
	f1_sp = np.max(f1_scores[np.isfinite(f1_scores)])

	precisions, recalls, _ = precision_recall_curve(gt_px.ravel(), pr_px.ravel())
	f1_scores = (2 * precisions * recalls) / (precisions + recalls + 1e-12)
	f1_px = np.max(f1_scores[np.isfinite(f1_scores)])

	if len(gt_px.shape) == 4:
		gt_px = gt_px.squeeze(1)
	if len(pr_px.shape) == 4:
		pr_px = pr_px.squeeze(1)

	aupro = cal_pro_score(gt_px, pr_px)

	row = [
		obj,
		str(np.round(auroc_px * 100, 1)),
		str(np.round(f1_px * 100, 1)),
		str(np.round(ap_px * 100, 1)),
		str(np.round(aupro * 100, 1)),
		str(np.round(auroc_sp * 100, 1)),
		str(np.round(f1_sp * 100, 1)),
		str(np.round(ap_sp * 100, 1)),
	]

	metrics = {
		"auroc_px": auroc_px,
		"f1_px": f1_px,
		"ap_px": ap_px,
		"aupro": aupro,
		"auroc_sp": auroc_sp,
		"f1_sp": f1_sp,
		"ap_sp": ap_sp,
	}

	return obj, row, metrics


def evaluate_metrics(results, obj_list, num_workers: int = 1):
	payloads = []

	for obj in tqdm(obj_list, desc="Preparing eval payloads", leave=True):
		gt_px, pr_px, gt_sp, pr_sp = [], [], [], []

		for idx in range(len(results["cls_names"])):
			if results["cls_names"][idx] == obj:
				mask = results["imgs_masks"][idx]

				if torch.is_tensor(mask):
					mask = mask.cpu().numpy()

				mask = np.squeeze(mask)

				gt_px.append(mask)
				pr_px.append(results["anomaly_maps"][idx])
				gt_sp.append(results["gt_sp"][idx])
				pr_sp.append(results["pr_sp"][idx])

		payloads.append((obj, gt_px, pr_px, gt_sp, pr_sp))

	if num_workers is None or num_workers <= 1:
		worker_outputs = []
		for payload in tqdm(payloads, desc="Evaluating objects", leave=True):
			worker_outputs.append(evaluate_one_object_metric(payload))
	else:
		max_workers = min(num_workers, len(payloads))

		worker_outputs = []
		with ProcessPoolExecutor(max_workers=max_workers) as executor:
			futures = [executor.submit(evaluate_one_object_metric, payload) for payload in payloads]

			for future in tqdm(
				as_completed(futures),
				total=len(futures),
				desc=f"Evaluating objects ({max_workers} workers)",
				leave=True
			):
				worker_outputs.append(future.result())

	output_by_obj = {
		obj: (row, metrics)
		for obj, row, metrics in worker_outputs
	}

	table_ls = []
	auroc_sp_ls, auroc_px_ls = [], []
	f1_sp_ls, f1_px_ls = [], []
	aupro_ls, ap_sp_ls, ap_px_ls = [], [], []

	for obj in obj_list:
		row, metrics = output_by_obj[obj]

		table_ls.append(row)

		auroc_px_ls.append(metrics["auroc_px"])
		f1_px_ls.append(metrics["f1_px"])
		ap_px_ls.append(metrics["ap_px"])
		aupro_ls.append(metrics["aupro"])
		auroc_sp_ls.append(metrics["auroc_sp"])
		f1_sp_ls.append(metrics["f1_sp"])
		ap_sp_ls.append(metrics["ap_sp"])

	table_ls.append([
		"mean",
		str(np.round(np.mean(auroc_px_ls) * 100, 1)),
		str(np.round(np.mean(f1_px_ls) * 100, 1)),
		str(np.round(np.mean(ap_px_ls) * 100, 1)),
		str(np.round(np.mean(aupro_ls) * 100, 1)),
		str(np.round(np.mean(auroc_sp_ls) * 100, 1)),
		str(np.round(np.mean(f1_sp_ls) * 100, 1)),
		str(np.round(np.mean(ap_sp_ls) * 100, 1)),
	])

	return tabulate(
		table_ls,
		headers=["objects", "auroc_px", "f1_px", "ap_px", "aupro", "auroc_sp", "f1_sp", "ap_sp"],
		tablefmt="pipe"
	)


def build_cached_text_prompt_batch(text_prompts, cls_names, device):
	features = []

	for cls in cls_names:
		if cls not in text_prompts:
			raise KeyError(f"text prompts for class '{cls}' not found")
		features.append(text_prompts[cls])

	return torch.stack(features, dim=0).to(device=device, non_blocking=True)


def build_target_transforms(image_size):
	target_transform_b = transforms.Compose([
		transforms.Resize((image_size, image_size)),
		transforms.CenterCrop(image_size),
		transforms.ToTensor()
	])

	target_transform_type = transforms.Compose([
		transforms.Resize((image_size, image_size), interpolation=InterpolationMode.NEAREST),
		transforms.CenterCrop(image_size),
		transforms.PILToTensor(),
		transforms.Lambda(lambda x: x.squeeze(0).long()),
	])

	return target_transform_b, target_transform_type


def compute_frozen_global_scores(args, text_prompts, device):
	"""Frozen CLIP zero-shot image score per image path, for --global_fusion_weight.

	The global embedding needs no patch grid, so it comes from a second, untouched
	CLIP at its native --global_image_size instead of the 518 px of the maps. The
	score is the anomaly probability 1 - softmax(g @ text / T)[normal], averaged
	over the image and its horizontal flip; it reproduces G336f of
	tools/global_native_scores.py --fp32 and tools/image_score_fusion.py. The pass
	runs in fp32: under fp16 autocast the embeddings shift with the batch size,
	which moves the fused MPDD image AUROC by about 0.7.
	"""
	g_model, _, g_preprocess = open_clip.create_model_and_transforms(
		args.model,
		args.global_image_size,
		pretrained=args.pretrained
	)
	g_model = g_model.to(device)
	g_model.eval()

	g_data = build_dataset(args, g_preprocess, *build_target_transforms(args.global_image_size))

	# the global pass has its own fp32 text anchors (the map's are built under fp16
	# autocast), matching tools/global_native_scores.py --fp32 and global_crop_scores.py
	with torch.inference_mode():
		text_prompts = build_text_prompts(
			g_model, g_data.get_cls_names(), open_clip.get_tokenizer(args.model), device, args.dataset)
	text_prompts = {
		k: F.normalize(v.to(device=device, dtype=torch.float32), dim=0).contiguous()
		for k, v in text_prompts.items()
	}
	g_loader = torch.utils.data.DataLoader(
		g_data,
		batch_size=args.test_batch_size,
		shuffle=False,
		num_workers=args.num_workers,
		pin_memory=(device == "cuda"),
	)

	scores = {}
	for items in tqdm(g_loader, desc=f"Frozen global @{args.global_image_size}", leave=True):
		images = items["img"].to(device, non_blocking=True)
		cls_names = list(items["cls_name"])
		prob = torch.zeros(images.shape[0], device=device)

		for x in (images, torch.flip(images, dims=[3])):
			with torch.inference_mode():
				g, _ = g_model.encode_image(x, args.features_list)
				g = F.normalize(g.float(), dim=-1)
				cos = [g[i] @ text_prompts[c] for i, c in enumerate(cls_names)]
			for i, c in enumerate(cos):		# class 0 of every product is "normal"
				prob[i] += (1.0 - torch.softmax(c.float() / args.global_fusion_temperature, dim=-1)[0]) / 2

		scores.update(zip(items["img_path"], prob.cpu().tolist()))

	# --global_crop_grid g: the same square at g * size px, cut into g x g tiles of
	# the native size and scored like the whole image; the most anomalous tile is
	# averaged with the whole-image score (tools/global_crop_scores.py offline)
	if args.global_crop_grid > 0:
		gr, ts = args.global_crop_grid, args.global_image_size
		c_data = build_dataset(args, image_transform(gr * ts, is_train=False), *build_target_transforms(gr * ts))
		c_loader = torch.utils.data.DataLoader(
			c_data, batch_size=args.test_batch_size, shuffle=False,
			num_workers=args.num_workers, pin_memory=(device == "cuda"))
		for items in tqdm(c_loader, desc=f"Frozen global tiles {gr}x{gr}", leave=True):
			images = items["img"].to(device, non_blocking=True)
			cls_names = list(items["cls_name"])
			B = images.shape[0]
			tiles = images.reshape(B, 3, gr, ts, gr, ts).permute(0, 2, 4, 1, 3, 5).reshape(B * gr * gr, 3, ts, ts)
			prob = torch.zeros(B, gr * gr, device=device)
			for x in (tiles, torch.flip(tiles, dims=[3])):
				with torch.inference_mode():
					g = torch.cat([g_model.encode_image(ch, args.features_list)[0]
								   for ch in x.split(args.test_batch_size)], 0)
					g = F.normalize(g.float(), dim=-1).reshape(B, gr * gr, -1)
					for i, c in enumerate(cls_names):
						prob[i] += (1.0 - torch.softmax((g[i] @ text_prompts[c]).float() / args.global_fusion_temperature, dim=-1)[:, 0]) / 2
			for path, m in zip(items["img_path"], prob.max(dim=1).values.cpu().tolist()):
				scores[path] = (scores[path] + m) / 2

	del g_model
	if device == "cuda":
		torch.cuda.empty_cache()

	return scores


def load_frozen_global_scores(path, temperature):
	"""The scores of compute_frozen_global_scores, from a saved
	tools/global_native_scores.py --fp32 run (g_cos and g_cos_flip per image path),
	which that function reproduces. Saves the CLIP pass when one dataset is scored
	with many checkpoints; a missing image path fails at the fusion step."""
	d = np.load(path)

	def prob(cos):
		z = np.where(np.isnan(cos), -np.inf, cos / temperature)
		z = z - z.max(axis=1, keepdims=True)
		e = np.exp(z)
		return 1.0 - e[:, 0] / e.sum(axis=1)

	p = (prob(d["g_cos"]) + prob(d["g_cos_flip"])) / 2
	return dict(zip(d["img_path"].tolist(), p.tolist()))


def add_cached_crop_scores(scores, path, temperature):
	"""Average the most anomalous tile from a saved tools/global_crop_scores.py run
	into the whole-image scores, as --global_crop_grid does online."""
	d = np.load(path)

	def prob(cos):
		z = np.where(np.isnan(cos), -np.inf, cos / temperature)
		z = z - z.max(axis=-1, keepdims=True)
		e = np.exp(z)
		return 1.0 - e[..., 0] / e.sum(axis=-1)

	tile = ((prob(d["g_crop_cos"]) + prob(d["g_crop_cos_flip"])) / 2).max(axis=1)
	for path_i, m in zip(d["img_path"].tolist(), tile.tolist()):
		scores[path_i] = (scores[path_i] + m) / 2
	return scores


def test(args):
	device = "cuda" if torch.cuda.is_available() else "cpu"
	use_amp = (device == "cuda")

	logger = build_logger(args.save_path)

	for arg in vars(args):
		logger.info(f"{arg}: {getattr(args, arg)}")

	model, _, preprocess = open_clip.create_model_and_transforms(
		args.model,
		args.image_size,
		pretrained=args.pretrained
	)
	model = model.to(device)
	model.eval()

	tokenizer = open_clip.get_tokenizer(args.model)

	target_transform_b, target_transform_type = build_target_transforms(args.image_size)

	test_data = build_dataset(args, preprocess, target_transform_b, target_transform_type)

	test_loader = torch.utils.data.DataLoader(
		test_data,
		batch_size=args.test_batch_size,
		shuffle=False,
		num_workers=args.num_workers,
		pin_memory=(device == "cuda"),
		persistent_workers=(args.num_workers > 0),
		prefetch_factor=args.prefetch_factor if args.num_workers > 0 else None,
	)

	obj_list = test_data.get_cls_names()

	with torch.inference_mode(), torch.cuda.amp.autocast(enabled=use_amp):
		text_prompts = build_text_prompts(
			model=model,
			obj_list=obj_list,
			tokenizer=tokenizer,
			device=device,
			dataset_name=args.dataset
		)

		semantic_embeddings, semantic_meta = get_averaged_text_prompt_embeddings(
			model=model,
			objs=obj_list,
			tokenizer=tokenizer,
			device=device,
			dataset_name=get_prompt_dataset_name(args.dataset),
			normalize=True,
			return_meta=True
		)

	text_prompts = {
		k: F.normalize(v.to(device=device, dtype=torch.float32), dim=0).contiguous()
		for k, v in text_prompts.items()
	}

	global_scores = None
	if args.global_fusion_weight > 0:
		if args.global_scores_npz:
			global_scores = load_frozen_global_scores(args.global_scores_npz, args.global_fusion_temperature)
			if args.global_crop_grid > 0:
				if not args.global_crop_npz:
					raise ValueError("--global_scores_npz with --global_crop_grid > 0 needs --global_crop_npz "
									 "(or --global_crop_grid 0 for the whole-image score alone)")
				global_scores = add_cached_crop_scores(global_scores, args.global_crop_npz, args.global_fusion_temperature)
		else:
			global_scores = compute_frozen_global_scores(args, text_prompts, device)
		logger.info(
			f"Frozen CLIP global score at {args.global_image_size} px for {len(global_scores)} images; "
			f"image score = {1 - args.global_fusion_weight:g} * map + {args.global_fusion_weight:g} * global."
		)

	logger.info(f"Built text prompts for {len(text_prompts)} products.")
	logger.info(f"Semantic embedding tensor shape: {tuple(semantic_embeddings.shape)}")

	checkpoint = torch.load(args.checkpoint_path, map_location=device)

	if "hybrid_codebook" not in checkpoint:
		raise KeyError(f"No 'hybrid_codebook' in checkpoint. Available keys: {list(checkpoint.keys())}")

	# checkpoints trained with --no_image_branch have no adaptor; only the
	# codebook-routed global score needs one
	if "image_adaptor" not in checkpoint and args.image_score_mode == "global":
		raise KeyError(
			"No 'image_adaptor' in checkpoint. "
			"--image_score_mode global needs a checkpoint trained with the global image adaptor."
		)

	hybrid_state = checkpoint["hybrid_codebook"]

	# train.py --per_layer_codebook saves one codebook per feature layer
	per_layer = bool(checkpoint.get("per_layer_codebook", False))
	if per_layer:
		saved_list = [hybrid_state[f"{l}.learnable_entries"] for l in range(len(args.features_list))]
	else:
		if "learnable_entries" not in hybrid_state:
			raise KeyError(
				f"'learnable_entries' not found in checkpoint['hybrid_codebook']. "
				f"Available keys: {list(hybrid_state.keys())}"
			)
		saved_list = [hybrid_state["learnable_entries"]]

	saved_learnable = saved_list[0]
	num_learnable = saved_learnable.shape[0]

	calib_entries = None
	if args.semantic_calibration or args.residual_diag:
		if args.calib_control == "random":
			# Same count, same norm, no semantics: isolates the effect of enlarging
			# the assignment distribution from the effect of what was added to it.
			g = torch.Generator(device="cpu").manual_seed(args.seed)
			calib_entries = F.normalize(
				torch.randn(semantic_embeddings.shape, generator=g), dim=-1
			).to(device=device, dtype=semantic_embeddings.dtype)
			logger.info(
				"CONTROL --calib_control random: %d random unit directions replace the "
				"text entries on the calibration path.", calib_entries.shape[0]
			)
		else:
			calib_entries = semantic_embeddings

	def make_codebook(entries):
		cb = HybridCodebook(
			semantic_embeddings=semantic_embeddings if args.with_semantic_codebook else None,
			calib_semantic_embeddings=calib_entries,
			num_learnable=entries.shape[0],
			embed_dim=semantic_embeddings.shape[-1],
			beta=args.beta
		).to(device)
		with torch.no_grad():
			cb.learnable_entries.copy_(entries.to(device))
		return cb.eval()

	# codebooks[l] quantizes feature layer l; with a shared codebook every entry
	# is the same module. hybrid_codebook names the last one for the diagnostics.
	codebooks = [make_codebook(e) for e in saved_list]
	if not per_layer:
		codebooks = codebooks * len(args.features_list)
	hybrid_codebook = codebooks[-1]
	if per_layer:
		logger.info(f"Per-layer codebooks: {len(codebooks)} x {num_learnable} entries.")

	if args.with_semantic_codebook:
		logger.info(
			f"ABLATION --with_semantic_codebook: appended target-vocabulary entries "
			f"with shape {tuple(semantic_embeddings.shape)}; these are selected for "
			f"well under 1%% of patches (see the usage line below)."
		)
	else:
		logger.info(
			"Quantizing against the trained prototypes only; no target vocabulary "
			"is added to the codebook."
		)

	if args.semantic_calibration:
		logger.info(
			f"CALIBRATION-PATH HYBRID: {semantic_embeddings.shape[0]} target-vocabulary "
			f"entries enter the assignment distribution only. Assignment, quantized "
			f"value and text-anchor scoring are unchanged; the entropy that sets the "
			f"per-pixel temperature is now taken over {num_learnable} prototypes plus "
			f"{semantic_embeddings.shape[0]} named concepts."
		)
	logger.info(
		f"Loaded only learnable_entries from checkpoint with shape "
		f"{tuple(saved_learnable.shape)}."
	)

	with open(args.config_path, "r") as f:
		model_configs = json.load(f)

	linearlayer = LinearLayer(
		model_configs["vision_cfg"]["width"],
		model_configs["embed_dim"],
		len(args.features_list),
		args.model
	).to(device)

	linearlayer.load_state_dict(checkpoint["trainable_linearlayer"])
	linearlayer.eval()

	image_adaptor = None
	if "image_adaptor" in checkpoint:
		image_adaptor = GlobalImageAdaptor(
			embed_dim=model_configs["embed_dim"],
			hidden_dim=args.image_adaptor_hidden_dim,
			dropout=args.image_adaptor_dropout
		).to(device)
		image_adaptor.load_state_dict(checkpoint["image_adaptor"])
		image_adaptor.eval()

	results = {
		"cls_names": [],
		"img_paths": [],
		"imgs_masks": [],
		"anomaly_maps": [],
		"gt_sp": [],
		"pr_sp": [],
	}

	# Optional per-image dump for the standalone AU-sPRO evaluator. Maps are kept
	# at model resolution as float16; the evaluator upsamples them to the raw GT
	# size (where the absolute saturation thresholds are defined).
	dump_records = defaultdict(list) if args.dump_maps else None
	map_scores = {}		# image path -> map term of the image score, for --dump_image_scores
	dump_root = os.path.join(args.save_path, "dump") if args.dump_maps else None

	# Diagnostic: how often the semantic half of the codebook is actually selected.
	# If this is ~0 the test-time extension is inert, whatever the metrics say.
	assign_sem = 0
	assign_total = 0
	sim_sem_sum = 0.0
	sim_learn_sum = 0.0

	# Residual-quantization diagnostic. Over anomalous patches only (gt mask = 1),
	# how often does the residual x - z_q land on a concept whose product matches
	# the image, and on one whose defect type matches the ground truth? Compared
	# against the chance rate for the same candidate set, this says whether the
	# residual carries recoverable semantic content at all.
	resid_prod_hit = 0
	resid_defect_hit = 0
	resid_total = 0
	resid_chance_prod = 0.0
	resid_chance_defect = 0.0
	if args.residual_diag:
		sem_products = [m["product"] for m in semantic_meta]
		sem_defects = [m["defect"] for m in semantic_meta]
		logger.info(
			f"RESIDUAL DIAGNOSTIC over {len(semantic_meta)} concepts spanning "
			f"{len(set(sem_products))} products and {len(set(sem_defects))} defect types."
		)

	for items in tqdm(test_loader, desc="Testing", total=len(test_loader), leave=True):
		images = items["img"].to(device, non_blocking=True)
		cls_names = list(items["cls_name"])
		img_paths = list(items["img_path"])

		gt_masks = items["img_mask_b"].clone()
		gt_masks[gt_masks > 0.5] = 1
		gt_masks[gt_masks <= 0.5] = 0

		with torch.inference_mode(), torch.cuda.amp.autocast(enabled=use_amp):
			image_features, patch_tokens = model.encode_image(images, args.features_list)

			patch_tokens = linearlayer(patch_tokens)
			if args.tta_flip:		# mirrored pass, averaged into the map below
				_, patch_tokens_flip = model.encode_image(torch.flip(images, dims=[3]), args.features_list)
				patch_tokens_flip = linearlayer(patch_tokens_flip)

			grouped_indices = defaultdict(list)
			for i, cls_name in enumerate(cls_names):
				grouped_indices[cls_name].append(i)

			anomaly_maps_gpu = [None] * images.shape[0]
			pr_sp_list = [None] * images.shape[0]

			for cls_name, indices in grouped_indices.items():
				idx_tensor = torch.tensor(indices, device=device, dtype=torch.long)

				image_features_g = image_features.index_select(0, idx_tensor)

				text_features_g = build_cached_text_prompt_batch(
					text_prompts=text_prompts,
					cls_names=[cls_name] * len(indices),
					device=device
				)

				# Image-level score.
				if args.image_score_mode == "global":
					# global CLIP image embedding -> image adaptor -> hybrid codebook -> text anchors.
					pr_sp_g = compute_global_image_score(
						image_features=image_features_g,
						text_features=text_features_g,
						image_adaptor=image_adaptor,
						hybrid_codebook=hybrid_codebook,
						temperature=args.image_temperature,
						assign_temperature=args.assign_temperature
					)

					for local_i, global_i in enumerate(indices):
						pr_sp_list[global_i] = pr_sp_g[local_i].detach().cpu().item()

				acc_map = None

				for layer_idx in range(len(patch_tokens)):
					layer_tokens = extract_patch_tokens_for_layer(
						patch_tokens[layer_idx].index_select(0, idx_tensor),
						args.model
					)

					ret = codebooks[layer_idx](
						layer_tokens,
						assign_temperature=args.assign_temperature,
						semantic_bias=args.semantic_bias,
						assign_mode=args.assign_mode
					)

					patch_q = ret["z_q_st"]
					assign_probs = ret["assign_probs"]

					if hybrid_codebook.num_semantic > 0:
						assign_sem += int((ret["indices"] < hybrid_codebook.num_semantic).sum().item())
						assign_total += int(ret["indices"].numel())
						# Unbiased similarity gap between the two halves, to size the
						# offset the semantic entries would need to be competitive.
						raw = ret["raw_logits"]
						sim_sem_sum += float(raw[..., :hybrid_codebook.num_semantic].max(dim=-1).values.sum().item())
						sim_learn_sum += float(raw[..., hybrid_codebook.num_semantic:].max(dim=-1).values.sum().item())

					if args.residual_diag and ret["resid_indices"] is not None:
						# Anomalous patches only: a residual on a normal patch has no
						# defect type to be right or wrong about. Ground truth comes
						# from the directory name of the test image, which is the
						# defect type in every dataset we evaluate.
						H_tok = int(np.sqrt(layer_tokens.shape[1]))
						gt_g = gt_masks.index_select(0, idx_tensor.cpu()).to(device).float()
						if gt_g.dim() == 3:
							gt_g = gt_g.unsqueeze(1)			# [B,1,H,W]
						gt_small = F.interpolate(
							gt_g, size=(H_tok, H_tok), mode="nearest"
						).squeeze(1).reshape(len(indices), -1) > 0.5
						if gt_small.any():
							ri = ret["resid_indices"][gt_small]
							paths_g = [img_paths[i] for i in indices]
							gt_defect = [
								os.path.basename(os.path.dirname(p)).lower() for p in paths_g
							]
							per_img = gt_small.sum(dim=1).tolist()
							expanded = []
							for n, d in zip(per_img, gt_defect):
								expanded.extend([d] * int(n))
							for k, idx_sem in enumerate(ri.tolist()):
								resid_total += 1
								if sem_products[idx_sem] == cls_name:
									resid_prod_hit += 1
								if sem_defects[idx_sem].lower() == expanded[k]:
									resid_defect_hit += 1
							# Chance rate for this image's candidate set.
							n_prod = sum(1 for p in sem_products if p == cls_name)
							resid_chance_prod += len(ri) * n_prod / len(sem_products)
							for d in expanded:
								n_def = sum(1 for s in sem_defects if s.lower() == d)
								resid_chance_defect += n_def / len(sem_defects)

					if args.score_blend > 0.0:
						# Blend the raw CLIP patch feature with the hard-quantized code
						# to retain the fine deviation that nearest-prototype
						# quantization discards (alpha=0 -> current, alpha=1 -> raw).
						raw_patch = F.normalize(layer_tokens, dim=-1)
						zq_patch = F.normalize(patch_q, dim=-1)
						patch_used = F.normalize(
							args.score_blend * raw_patch + (1.0 - args.score_blend) * zq_patch,
							dim=-1
						)
					else:
						patch_used = F.normalize(patch_q, dim=-1)

					# Raw similarity logits, not divided by temperature yet.
					sim_logits = torch.bmm(patch_used, text_features_g)

					Bg, L, C = sim_logits.shape
					H = int(np.sqrt(L))

					if H * H != L:
						raise ValueError(f"L={L} is not square.")

					sim_logits = F.interpolate(
						sim_logits.permute(0, 2, 1).contiguous().view(Bg, C, H, H),
						size=(args.image_size, args.image_size),
						mode="bilinear",
						align_corners=True
					)

					if args.cali:
						calib_map = build_calibration_map(
							assign_probs=assign_probs,
							img_size=args.image_size
						)

						seg_prob = entropy_adaptive_softmax(
							sim_logits=sim_logits,
							calib_map=calib_map,
							min_temperature=args.temperature,
							max_temperature=args.max_temperature
						)
					else:
						seg_prob = torch.softmax(sim_logits / args.temperature, dim=1)

					layer_map = seg_prob[:, 1:, :, :].sum(dim=1)

					acc_map = layer_map if acc_map is None else acc_map + layer_map

				acc_map = acc_map / len(patch_tokens)
				if args.tta_flip:
					acc_flip = anomaly_map_from_tokens(patch_tokens_flip, idx_tensor, text_features_g, codebooks, args)
					acc_map = 0.5 * (acc_map + torch.flip(acc_flip, dims=[-1]))
				acc_map = gaussian_blur_maps(acc_map, args.smooth_sigma)

				# Image-level score from top-k mean of the pixel anomaly map. With several
				# ratios the top-k means are averaged (multi-scale read-out), so small
				# and large defects both reach the score without choosing one ratio.
				if args.image_score_mode == "topk_map":
					Bacc = acc_map.shape[0]
					flat_map = acc_map.reshape(Bacc, -1)
					img_score = torch.stack([
						torch.topk(flat_map, max(1, int(round(r * flat_map.shape[1]))), dim=1).values.mean(dim=1).float()
						for r in args.topk_ratio
					], dim=1).mean(dim=1)
					for local_i, global_i in enumerate(indices):
						s = img_score[local_i].detach().cpu().item()
						map_scores[img_paths[global_i]] = s
						if global_scores is not None:
							w = args.global_fusion_weight
							s = (1.0 - w) * s + w * global_scores[img_paths[global_i]]
						pr_sp_list[global_i] = s

				for local_i, global_i in enumerate(indices):
					anomaly_maps_gpu[global_i] = acc_map[local_i]

		for b in range(images.shape[0]):
			cls_name = cls_names[b]
			img_path = img_paths[b]

			anomaly_map_np = anomaly_maps_gpu[b].float().cpu().numpy()

			results["cls_names"].append(cls_name)
			results["img_paths"].append(img_path)
			results["imgs_masks"].append(gt_masks[b:b+1])
			results["gt_sp"].append(items["anomaly"][b].item())
			results["pr_sp"].append(pr_sp_list[b])
			results["anomaly_maps"].append(anomaly_map_np)

			if args.save_vis:
				save_visualization(
					image_path=img_path,
					anomaly_map=anomaly_map_np[None, ...],
					img_size=args.image_size,
					save_path=args.save_path,
					cls_name=cls_name
				)

			if dump_records is not None:
				# defect_cls is the state subdir (good / logical_anomalies /
				# structural_anomalies for LOCO), which the evaluator uses for
				# the logical-vs-structural split. Not every loader supplies
				# it: MPDDDataset calls the same thing specie_name, and
				# VisaDatasetTest emits no defect label at all, so fall back in
				# that order and finally to the binary anomaly flag. Only the
				# read-out study consumes this, and it uses the field purely as
				# a path component, so the fallback costs it nothing.
				if "defect_cls" in items:
					defect_cls = items["defect_cls"][b]
				elif "specie_name" in items:
					defect_cls = items["specie_name"][b]
				else:
					defect_cls = "anomaly" if int(items["anomaly"][b].item()) else "good"
				stem = os.path.splitext(os.path.basename(img_path))[0]
				out_dir = os.path.join(dump_root, cls_name, defect_cls)
				os.makedirs(out_dir, exist_ok=True)
				dumped = anomaly_map_np
				if args.dump_stride > 1:
					dumped = dumped[::args.dump_stride, ::args.dump_stride]
				np.save(os.path.join(out_dir, stem + ".npy"),
						dumped.astype(np.float16))
				dump_records[cls_name].append({
					"stem": stem,
					"defect_cls": defect_cls,
					"anomaly": int(items["anomaly"][b].item()),
					"image_score": float(pr_sp_list[b]),
					"img_path": img_path,
				})

	if dump_records is not None:
		for cls_name, recs in dump_records.items():
			with open(os.path.join(dump_root, cls_name, "scores.json"), "w") as f:
				json.dump(recs, f)
		logger.info("dumped per-image maps for %d classes to %s",
					len(dump_records), dump_root)

	if assign_total > 0:
		logger.info(
			"codebook usage: %.3f%% of patch assignments went to a semantic entry "
			"(%d / %d), with %d semantic and %d learnable entries, mode=%s, bias=%.3f.",
			100.0 * assign_sem / assign_total, assign_sem, assign_total,
			hybrid_codebook.num_semantic, hybrid_codebook.num_learnable,
			args.assign_mode, args.semantic_bias
		)
		logger.info(
			"unbiased mean best cosine: semantic %.4f, learnable %.4f, gap %.4f.",
			sim_sem_sum / assign_total, sim_learn_sum / assign_total,
			(sim_learn_sum - sim_sem_sum) / assign_total
		)
	else:
		logger.info("codebook usage: vocabulary-free codebook, no semantic entries present.")

	if resid_total > 0:
		logger.info(
			"residual diagnostic: %d anomalous patches. product match %.2f%% "
			"(chance %.2f%%), defect-type match %.2f%% (chance %.2f%%).",
			resid_total,
			100.0 * resid_prod_hit / resid_total,
			100.0 * resid_chance_prod / resid_total,
			100.0 * resid_defect_hit / resid_total,
			100.0 * resid_chance_defect / resid_total,
		)

	if args.dump_image_scores:
		paths = results["img_paths"]
		np.savez(
			os.path.join(args.save_path, "image_scores.npz"),
			img_path=np.array(paths), cls_name=np.array(results["cls_names"]),
			label=np.array(results["gt_sp"]), score=np.array(results["pr_sp"], dtype=np.float64),
			map_score=np.array([map_scores.get(p, np.nan) for p in paths], dtype=np.float64),
			global_score=np.array([global_scores[p] if global_scores is not None else np.nan for p in paths], dtype=np.float64),
		)

	table_str = evaluate_metrics(
		results=results,
		obj_list=obj_list,
		num_workers=args.eval_workers
	)

	logger.info("\n%s", table_str)


if __name__ == "__main__":
	parser = argparse.ArgumentParser("Hybrid Codebook Test", add_help=True)

	# paths
	parser.add_argument("--data_path", type=str, default="./data/visa", help="path to test dataset")
	parser.add_argument("--save_path", type=str, default="./results/test_clean", help="path to save results")
	parser.add_argument("--checkpoint_path", type=str, required=True, help="path to checkpoint")
	parser.add_argument("--config_path", type=str, default="./open_clip/model_configs/ViT-L-14-336.json", help="model configs")

	# model
	parser.add_argument("--dataset", type=str, default="visa", help="test dataset")
	parser.add_argument("--model", type=str, default="ViT-L-14-336", help="CLIP model name")
	parser.add_argument("--pretrained", type=str, default="openai", help="CLIP pretrained weights")
	parser.add_argument("--features_list", type=int, nargs="+", default=[6, 12, 18, 24], help="feature layers")
	parser.add_argument("--image_size", type=int, default=518, help="image size")

	# pixel-level temperature
	parser.add_argument("--temperature", type=float, default=0.01, help="minimum/base temperature for pixel-level scoring")
	parser.add_argument("--max_temperature", type=float, default=0.1, help="maximum temperature for entropy adaptive pixel calibration")

	# image-level score
	parser.add_argument("--image_temperature", type=float, default=0.1, help="temperature for global image-level anomaly score")
	parser.add_argument("--image_adaptor_hidden_dim", type=int, default=None, help="hidden dim of image adaptor, None means embed_dim")
	parser.add_argument("--image_adaptor_dropout", type=float, default=0.1, help="dropout for image adaptor architecture")
	parser.add_argument("--image_score_mode", type=str, default="topk_map", choices=["global", "topk_map"], help="image-level anomaly score source: 'global' branch or 'topk_map' pooling of the pixel anomaly map")
	parser.add_argument("--topk_ratio", type=float, nargs="+", default=[0.0005, 0.001, 0.005, 0.01, 0.05, 0.1], help="fraction(s) of top pixels averaged for the topk_map image score (k>=1); with several values the score is the mean of the top-k means (multi-scale read-out). Default: the grid of the paper; 0.001 alone gives the earlier single-fraction read-out")
	parser.add_argument("--global_fusion_weight", type=float, default=0.75, help="weight w of the frozen CLIP zero-shot image score in the topk_map image score, (1-w)*map + w*global; 0 disables (map-only image score)")
	parser.add_argument("--global_fusion_temperature", type=float, default=0.1, help="softmax temperature of the frozen CLIP global score")
	parser.add_argument("--global_scores_npz", type=str, default="", help="load the frozen global scores from a tools/global_native_scores.py --fp32 output instead of recomputing them")
	parser.add_argument("--global_crop_grid", type=int, default=3, help="also score g x g tiles of the image at the native size and average the most anomalous tile into the global score; 0 disables (read-out B of the earlier draft)")
	parser.add_argument("--global_crop_npz", type=str, default="", help="with --global_scores_npz: load the tile scores from a tools/global_crop_scores.py output")
	parser.add_argument("--global_image_size", type=int, default=336, help="input size of the frozen CLIP that computes the global score (its native resolution)")
	parser.add_argument("--split", type=str, default="test", help="dataset split to evaluate; use 'valid' to tune hyperparameters off-test (bmad only)")
	parser.add_argument("--score_blend", type=float, default=0.0, help="pixel scoring: convex blend alpha*raw_patch + (1-alpha)*z_q before text similarity (0 = hard quantization, the paper; 1 = raw CLIP feature)")

	# runtime
	parser.add_argument("--seed", type=int, default=42, help="random seed")
	parser.add_argument("--num_workers", type=int, default=4, help="dataloader workers")
	parser.add_argument("--prefetch_factor", type=int, default=2, help="dataloader prefetch factor")
	parser.add_argument("--test_batch_size", type=int, default=8, help="test batch size")
	parser.add_argument("--save_vis", action="store_true", help="save anomaly visualization")
	parser.add_argument("--dump_image_scores", action="store_true", help="save per-image map, global and fused image scores to save_path/image_scores.npz")
	parser.add_argument("--dump_maps", action="store_true", help="dump per-image anomaly maps (float16) + image scores under <save_path>/dump for the standalone AU-sPRO evaluator (tools/loco_spro_eval.py)")
	parser.add_argument("--dump_stride", type=int, default=1, help="subsample dumped maps by this factor before writing; 1 keeps full resolution (required for AU-sPRO), 4 is enough for offline read-out studies and cuts the dump 16x")
	parser.add_argument("--fast", action="store_true", help="enable faster CUDA settings")

	# codebook
	parser.add_argument("--beta", type=float, default=0.1, help="commitment loss weight inside hybrid codebook")
	parser.add_argument("--assign_temperature", type=float, default=0.1, help="soft assignment temperature")
	parser.add_argument("--smooth_sigma", type=float, default=4.0, help="Gaussian sigma applied to the anomaly map before scoring and evaluation (4 in the paper); 0 disables")
	parser.add_argument("--assign_mode", type=str, default="raw", choices=["raw", "center", "zscore"], help="how the two halves of the codebook compete at assignment time: raw cosine, per-half mean removal (cancels the modality-gap offset), or per-half standardized similarity (competes on rank)")
	parser.add_argument("--semantic_bias", type=float, default=0.0, help="additive offset on the semantic logits at assignment time; compensates the modality gap that otherwise keeps the semantic half from ever being the argmax (0 = off)")
	parser.add_argument("--with_semantic_codebook", action="store_true", help="ablation (Sec. 'Does the codebook need the target vocabulary?'): append frozen target-vocabulary entries to the codebook. Off by default: they win <0.05%% of assignments and change no metric")
	parser.add_argument("--semantic_calibration", action="store_true", help="calibration-path hybrid: target-vocabulary entries enter ONLY the assignment distribution whose entropy sets the per-pixel temperature. They are not in the codebook, never win an argmax and never become the quantized value, so coverage asks 'known prototype OR named concept?' while scoring stays image-manifold")
	parser.add_argument("--residual_diag", action="store_true", help="diagnostic for residual quantization: re-quantize the residual (x - z_q) against the target vocabulary alone and report how often the winning concept matches the ground-truth defect type, against the chance rate. Requires --semantic_calibration to supply the entries")
	parser.add_argument("--calib_control", type=str, default="text", choices=["text", "random"], help="control for --semantic_calibration. Adding N near-zero-probability entries lowers the normalized entropy mechanically, because it is divided by log(K+N); 'random' substitutes N random unit directions for the text entries so that any gain from 'text' can be attributed to semantics rather than to the count")

	# pixel calibration
	parser.add_argument("--cali", action=argparse.BooleanOptionalAction, default=True, help="entropy-adaptive temperature for the pixel map (on in the paper); --no-cali scores every pixel at --temperature")
	parser.add_argument("--tta_flip", action="store_true", help="average the anomaly map with that of the horizontally flipped image (flipped back) before smoothing")

	# evaluation
	parser.add_argument("--eval_workers", type=int, default=4, help="number of processes for metric evaluation")

	args = parser.parse_args()
	if args.global_fusion_weight > 0 and args.image_score_mode != "topk_map":
		parser.error("--global_fusion_weight needs --image_score_mode topk_map")

	if args.fast:
		setup_speed(args.seed)
	else:
		setup_seed(args.seed)

	test(args)