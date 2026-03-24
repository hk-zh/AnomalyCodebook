import os
import cv2
import json
import torch
import random
import logging
import argparse
import numpy as np
from skimage import measure
from tabulate import tabulate
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms

from sklearn.metrics import (
	auc,
	roc_auc_score,
	average_precision_score,
	precision_recall_curve
)
from sklearn.metrics import pairwise

import open_clip

from few_shot import memory
from model import LinearLayer, HybridCodebook
from dataset import VisaDataset, MVTecDataset, MPDDDataset, MADDataset, RealIADDataset_v2

from prompt_ensemble_visa_19cls_test import encode_text_with_prompt_ensemble as encode_text_with_prompt_ensemble_visa
from prompt_ensemble_mvtec_20cls import encode_text_with_prompt_ensemble as encode_text_with_prompt_ensemble_mvtec
from new_prompt_ensemble_mpdd import encode_text_with_prompt_ensemble as encode_text_with_prompt_ensemble_mpdd
from prompt_ensemble_mad_sim import encode_text_with_prompt_ensemble as encode_text_with_prompt_ensemble_mad_sim
from prompt_ensemble_mad_real import encode_text_with_prompt_ensemble as encode_text_with_prompt_ensemble_mad_real
from prompt_ensemble_real_IAD_simple import encode_text_with_prompt_ensemble as encode_text_with_prompt_ensemble_real_iad

from tqdm import tqdm


def setup_seed(seed: int) -> None:
	torch.manual_seed(seed)
	torch.cuda.manual_seed_all(seed)
	np.random.seed(seed)
	random.seed(seed)
	torch.backends.cudnn.deterministic = True
	torch.backends.cudnn.benchmark = False


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


def build_logger(save_path: str) -> logging.Logger:
	os.makedirs(save_path, exist_ok=True)
	log_path = os.path.join(save_path, "log.txt")

	root_logger = logging.getLogger()
	for handler in root_logger.handlers[:]:
		root_logger.removeHandler(handler)
	root_logger.setLevel(logging.WARNING)

	logger = logging.getLogger("test")
	logger.setLevel(logging.INFO)

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


def build_dataset(args, preprocess, transform):
	if args.dataset == "mvtec":
		return MVTecDataset(root=args.data_path, transform=preprocess, target_transform=transform, aug_rate=-1, mode="test")
	if args.dataset == "visa":
		return VisaDataset(root=args.data_path, transform=preprocess, target_transform=transform, mode="test")
	if args.dataset == "mpdd":
		return MPDDDataset(root=args.data_path, transform=preprocess, target_transform=transform, aug_rate=-1, mode="test")
	if args.dataset in ["mad_sim", "mad_real"]:
		return MADDataset(root=args.data_path, transform=preprocess, target_transform=transform, mode="test")
	if args.dataset == "real_iad":
		return RealIADDataset_v2(root=args.data_path, transform=preprocess, target_transform=transform, aug_rate=-1, mode="test")
	raise ValueError(f"Unsupported dataset: {args.dataset}")


def build_text_prompts(args, model, obj_list, tokenizer, device):
	if args.dataset == "mvtec":
		return encode_text_with_prompt_ensemble_mvtec(model, obj_list, tokenizer, device)
	if args.dataset == "visa":
		return encode_text_with_prompt_ensemble_visa(model, obj_list, tokenizer, device)
	if args.dataset == "mpdd":
		return encode_text_with_prompt_ensemble_mpdd(model, obj_list, tokenizer, device)
	if args.dataset == "mad_sim":
		return encode_text_with_prompt_ensemble_mad_sim(model, obj_list, tokenizer, device)
	if args.dataset == "mad_real":
		return encode_text_with_prompt_ensemble_mad_real(model, obj_list, tokenizer, device)
	if args.dataset == "real_iad":
		return encode_text_with_prompt_ensemble_real_iad(model, obj_list, tokenizer, device)
	raise ValueError(f"Unsupported dataset: {args.dataset}")


def build_semantic_embeddings_from_text_prompts(text_prompts: dict, device: str) -> torch.Tensor:
	"""
	Build frozen semantic codebook entries from prompt features.

	Assumption:
		text_prompts[cls] is either [K, D] or [D, K]
	We convert each class to one semantic prototype by averaging prompt variants.
	"""
	semantic_embeddings = []
	for _, feat in text_prompts.items():
		if not torch.is_tensor(feat):
			raise TypeError("text prompt features must be torch.Tensor")
		feat = feat.to(device).float()

		if feat.ndim == 1:
			cls_feat = feat
		elif feat.ndim == 2:
			# convert to [K, D] if needed
			if feat.shape[0] > feat.shape[1]:
				# likely [K, D]
				cls_feat = feat.mean(dim=0)
			else:
				# likely [D, K]
				cls_feat = feat.mean(dim=1)
		else:
			raise ValueError(f"Unsupported text prompt shape: {feat.shape}")

		cls_feat = F.normalize(cls_feat, dim=-1)
		semantic_embeddings.append(cls_feat)

	return torch.stack(semantic_embeddings, dim=0)


def build_text_feature_batch(cls_name_list, text_prompts, device):
	"""
	Return text features in shape [B, D, K].
	"""
	text_features = []
	for cls in cls_name_list:
		feat = text_prompts[cls].to(device).float()

		if feat.ndim != 2:
			raise ValueError(f"text prompt feature must be 2D, got {feat.shape}")

		# convert to [D, K]
		if feat.shape[0] > feat.shape[1]:
			# [K, D] -> [D, K]
			feat = feat.transpose(0, 1)

		feat = F.normalize(feat, dim=0)
		text_features.append(feat)

	return torch.stack(text_features, dim=0)


def tokens_to_anomaly_map(tokens, text_features, img_size):
	"""
	tokens: [B, L, D]
	text_features: [B, D, K]
	return: [B, H, W]
	"""
	tokens = F.normalize(tokens, dim=-1)
	logits = (tokens @ text_features) * 100.0  # [B, L, K]

	B, L, K = logits.shape
	H = int(np.sqrt(L))
	assert H * H == L, f"L={L} is not square. Check if CLS token was removed."

	logits = logits.permute(0, 2, 1).contiguous().view(B, K, H, H)
	logits = F.interpolate(logits, size=img_size, mode="bilinear", align_corners=True)
	prob = torch.softmax(logits, dim=1)

	# abnormal score = sum of all abnormal channels
	return torch.sum(prob[:, 1:, :, :], dim=1)


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


def compute_few_shot_map(args, model, image, cls_name, mem_features, few_shot_features, img_size):
	_, patch_tokens = model.encode_image(image, few_shot_features)
	anomaly_maps_few_shot = []

	for idx, p in enumerate(patch_tokens):
		if "ViT" in args.model:
			p = p[0, 1:, :]
		else:
			p = p[0].view(p.shape[1], -1).permute(1, 0).contiguous()

		cos = pairwise.cosine_similarity(mem_features[cls_name][idx].cpu(), p.cpu())
		height = int(np.sqrt(cos.shape[1]))
		anomaly_map_few_shot = np.min((1 - cos), axis=0).reshape(1, 1, height, height)
		anomaly_map_few_shot = F.interpolate(
			torch.tensor(anomaly_map_few_shot),
			size=img_size,
			mode="bilinear",
			align_corners=True
		)
		anomaly_maps_few_shot.append(anomaly_map_few_shot[0].cpu().numpy())

	return np.sum(anomaly_maps_few_shot, axis=0)


def evaluate_metrics(results, obj_list):
	table_ls = []
	auroc_sp_ls, auroc_px_ls = [], []
	f1_sp_ls, f1_px_ls = [], []
	aupro_ls, ap_sp_ls, ap_px_ls = [], [], []

	for obj in obj_list:
		gt_px, pr_px, gt_sp, pr_sp, pr_sp_tmp = [], [], [], [], []

		for idx in range(len(results["cls_names"])):
			if results["cls_names"][idx] == obj:
				gt_px.append(results["imgs_masks"][idx].squeeze(1).numpy())
				pr_px.append(results["anomaly_maps"][idx])
				pr_sp_tmp.append(np.max(results["anomaly_maps"][idx]))
				gt_sp.append(results["gt_sp"][idx])
				pr_sp.append(results["pr_sp"][idx])

		gt_px = np.array(gt_px)
		pr_px = np.array(pr_px)
		gt_sp = np.array(gt_sp)
		pr_sp = np.array(pr_sp)

		pr_sp_tmp = np.array(pr_sp_tmp)
		if len(pr_sp_tmp) > 0:
			den = pr_sp_tmp.max() - pr_sp_tmp.min()
			pr_sp = (pr_sp_tmp - pr_sp_tmp.min()) / den if den > 1e-12 else np.zeros_like(pr_sp_tmp)

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

		table_ls.append([
			obj,
			str(np.round(auroc_px * 100, 1)),
			str(np.round(f1_px * 100, 1)),
			str(np.round(ap_px * 100, 1)),
			str(np.round(aupro * 100, 1)),
			str(np.round(auroc_sp * 100, 1)),
			str(np.round(f1_sp * 100, 1)),
			str(np.round(ap_sp * 100, 1)),
		])

		auroc_sp_ls.append(auroc_sp)
		auroc_px_ls.append(auroc_px)
		f1_sp_ls.append(f1_sp)
		f1_px_ls.append(f1_px)
		aupro_ls.append(aupro)
		ap_sp_ls.append(ap_sp)
		ap_px_ls.append(ap_px)

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


def test(args):
	device = "cuda" if torch.cuda.is_available() else "cpu"
	if torch.cuda.is_available():
		torch.backends.cudnn.benchmark = True

	logger = build_logger(args.save_path)
	for arg in vars(args):
		if args.mode == "zero_shot" and arg in ["k_shot", "few_shot_features"]:
			continue
		logger.info(f"{arg}: {getattr(args, arg)}")

	# CLIP
	model, _, preprocess = open_clip.create_model_and_transforms(
		args.model,
		args.image_size,
		pretrained=args.pretrained
	)
	model.to(device)
	model.eval()

	tokenizer = open_clip.get_tokenizer(args.model)

	# Config and trainable layer
	with open(args.config_path, "r") as f:
		model_configs = json.load(f)

	linearlayer = LinearLayer(
		model_configs["vision_cfg"]["width"],
		model_configs["embed_dim"],
		len(args.features_list),
		args.model
	).to(device)
	linearlayer.eval()

	# Dataset
	transform = transforms.Compose([
		transforms.Resize((args.image_size, args.image_size)),
		transforms.CenterCrop(args.image_size),
		transforms.ToTensor()
	])

	test_data = build_dataset(args, preprocess, transform)
	test_loader = torch.utils.data.DataLoader(
		test_data,
		batch_size=1,
		shuffle=False,
		num_workers=args.num_workers,
		pin_memory=True
	)
	obj_list = test_data.get_cls_names()

	# Build prompts for testing dataset
	with torch.inference_mode(), torch.cuda.amp.autocast(enabled=torch.cuda.is_available()):
		text_prompts = build_text_prompts(args, model, obj_list, tokenizer, device)
		semantic_embeddings = build_semantic_embeddings_from_text_prompts(text_prompts, device)

	# Hybrid codebook for testing dataset:
	# frozen semantic entries come from NEW test prompts
	# learnable entries come from checkpoint
	hybrid_codebook = HybridCodebook(
		semantic_embeddings=semantic_embeddings,
		num_learnable=args.codebook_num_learnable,
		embed_dim=semantic_embeddings.shape[-1]
	).to(device)
	hybrid_codebook.eval()

	# Checkpoint
	checkpoint = torch.load(args.checkpoint_path, map_location=device)
	linearlayer.load_state_dict(checkpoint["trainable_linearlayer"])

	if "hybrid_codebook" in checkpoint and "learnable_entries" in checkpoint["hybrid_codebook"]:
		saved_learnable = checkpoint["hybrid_codebook"]["learnable_entries"]
		assert saved_learnable.shape == hybrid_codebook.learnable_entries.shape, \
			f"learnable_entries shape mismatch: ckpt {saved_learnable.shape}, current {hybrid_codebook.learnable_entries.shape}"
		hybrid_codebook.learnable_entries.data.copy_(saved_learnable.to(device))
		logger.info("Loaded learnable entries from checkpoint.")
	else:
		logger.warning("No learnable_entries found in checkpoint. Using randomly initialized learnable tokens.")

	# Few-shot memory
	if args.mode == "few_shot":
		mem_features = memory(
			args.model,
			model,
			obj_list,
			args.data_path,
			args.save_path,
			preprocess,
			transform,
			args.k_shot,
			args.few_shot_features,
			args.dataset,
			device
		)

	results = {
		"cls_names": [],
		"imgs_masks": [],
		"anomaly_maps": [],
		"gt_sp": [],
		"pr_sp": [],
	}

	for items in tqdm(test_loader):
		image = items["img"].to(device, non_blocking=True)
		cls_name = items["cls_name"]
		img_path = items["img_path"][0]

		results["cls_names"].append(cls_name[0])

		gt_mask = items["img_mask"].clone()
		gt_mask[gt_mask > 0.5] = 1
		gt_mask[gt_mask <= 0.5] = 0

		results["imgs_masks"].append(gt_mask)
		results["gt_sp"].append(items["anomaly"].item())

		with torch.inference_mode(), torch.cuda.amp.autocast(enabled=torch.cuda.is_available()):
			image_features, patch_tokens = model.encode_image(image, args.features_list)
			image_features = F.normalize(image_features, dim=-1)

			text_features = build_text_feature_batch(cls_name, text_prompts, device)  # [B, D, K]

			# image-level score
			text_probs = ((image_features @ text_features[0]) * 100.0).softmax(dim=-1)
			results["pr_sp"].append(torch.sum(text_probs[0][1:]).cpu().item())

			# pixel-level score with codebook quantization
			patch_tokens = linearlayer(patch_tokens)
			layer_maps = []

			for layer_idx in range(len(patch_tokens)):
				ret = hybrid_codebook(patch_tokens[layer_idx])

				# inference: use quantized feature directly
				patch_q = ret["z_q"]
				layer_map = tokens_to_anomaly_map(patch_q, text_features, args.image_size)
				layer_maps.append(layer_map.cpu().numpy())

			anomaly_map = np.sum(layer_maps, axis=0)

			if args.mode == "few_shot":
				anomaly_map_few_shot = compute_few_shot_map(
					args=args,
					model=model,
					image=image,
					cls_name=cls_name[0],
					mem_features=mem_features,
					few_shot_features=args.few_shot_features,
					img_size=args.image_size
				)
				anomaly_map = anomaly_map + anomaly_map_few_shot

		results["anomaly_maps"].append(anomaly_map)

		if args.save_vis:
			save_visualization(
				image_path=img_path,
				anomaly_map=anomaly_map,
				img_size=args.image_size,
				save_path=args.save_path,
				cls_name=cls_name[0]
			)

	table_str = evaluate_metrics(results, obj_list)
	logger.info("\n%s", table_str)


if __name__ == "__main__":
	parser = argparse.ArgumentParser("Hybrid Codebook Test", add_help=True)

	parser.add_argument("--data_path", type=str, default="./data/visa", help="path to test dataset")
	parser.add_argument("--save_path", type=str, default="./results/test_clean", help="path to save results")
	parser.add_argument("--checkpoint_path", type=str, required=True, help="path to checkpoint")
	parser.add_argument("--config_path", type=str, default="./open_clip/model_configs/ViT-B-16.json", help="model config path")

	parser.add_argument("--dataset", type=str, default="mvtec", help="test dataset")
	parser.add_argument("--model", type=str, default="ViT-B-16", help="CLIP model name")
	parser.add_argument("--pretrained", type=str, default="laion400m_e32", help="CLIP pretrained weights")
	parser.add_argument("--features_list", type=int, nargs="+", default=[3, 6, 9], help="feature layers")
	parser.add_argument("--few_shot_features", type=int, nargs="+", default=[3, 6, 9], help="few-shot feature layers")
	parser.add_argument("--image_size", type=int, default=224, help="image size")
	parser.add_argument("--mode", type=str, default="zero_shot", choices=["zero_shot", "few_shot"], help="test mode")

	parser.add_argument("--codebook_num_learnable", type=int, default=64, help="number of learnable codebook tokens")
	parser.add_argument("--k_shot", type=int, default=10, help="few-shot K")
	parser.add_argument("--seed", type=int, default=10, help="random seed")
	parser.add_argument("--num_workers", type=int, default=4, help="dataloader workers")
	parser.add_argument("--save_vis", action="store_true", help="save anomaly visualization")

	args = parser.parse_args()

	setup_seed(args.seed)
	test(args)