# Copyright (c) 2025 Robert Bosch GmbH
# SPDX-License-Identifier: AGPL-3.0

import os
import json
import random
import logging
import argparse
import numpy as np

import torch
torch.autograd.set_detect_anomaly(True)
import torch.nn.functional as F
import torchvision.transforms as transforms
from torchvision.transforms import InterpolationMode

import open_clip

from tqdm import tqdm

from dataset import VisaDatasetV2, MVTecDataset, MPDDDataset, RealIADDataset_v2
from model import LinearLayer, HybridCodebook
from loss import FocalLoss, BinaryDiceLoss

from prompts.prompt_ensemble_mvtec_object_agnostic import encode_text_with_prompt_ensemble as encode_text_with_prompt_ensemble_mvtec
from prompts.prompt_ensemble_visa_19cls import encode_text_with_prompt_ensemble as encode_text_with_prompt_ensemble_visa
from prompts.prompt_ensemble_mpdd_object_agnostic import encode_text_with_prompt_ensemble as encode_text_with_prompt_ensemble_mpdd
from prompts.prompt_ensemble_real_IAD_simple import encode_text_with_prompt_ensemble as encode_text_with_prompt_ensemble_real_iad

from utils import print_model_stats
from test import gaussian_blur_maps


class GlobalImageAdaptor(torch.nn.Module):
	"""
	Trainable adaptor for frozen CLIP global image features.

	CLIP global image feature:
		[B, D]
	-> residual adaptor:
		[B, D]
	-> HybridCodebook:
		[B, 1, D]
	-> quantized global feature:
		[B, D]
	-> global anomaly score.
	"""

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


def setup_seed(seed):
	torch.manual_seed(seed)
	torch.cuda.manual_seed_all(seed)
	np.random.seed(seed)
	random.seed(seed)
	torch.backends.cudnn.deterministic = True
	torch.backends.cudnn.benchmark = False


def build_logger(save_path):
	os.makedirs(save_path, exist_ok=True)
	txt_path = os.path.join(save_path, "log.txt")

	root_logger = logging.getLogger()
	for handler in root_logger.handlers[:]:
		root_logger.removeHandler(handler)
	root_logger.setLevel(logging.WARNING)

	logger = logging.getLogger("train")
	logger.handlers.clear()
	logger.setLevel(logging.INFO)
	logger.propagate = False

	formatter = logging.Formatter(
		"%(asctime)s.%(msecs)03d - %(levelname)s: %(message)s",
		datefmt="%y-%m-%d %H:%M:%S"
	)

	file_handler = logging.FileHandler(txt_path, mode="w")
	file_handler.setFormatter(formatter)
	logger.addHandler(file_handler)

	console_handler = logging.StreamHandler()
	console_handler.setFormatter(formatter)
	logger.addHandler(console_handler)

	return logger


def build_train_dataset(args, preprocess, target_transform_b, target_transform_type):
	if args.dataset == "mvtec":
		return MVTecDataset(
			root=args.train_data_path,
			transform=preprocess,
			target_transform=target_transform_b,
			target_transform_type=target_transform_type,
			aug_rate=args.aug_rate
		)

	if args.dataset == "visa":
		return VisaDatasetV2(
			root=args.train_data_path,
			transform=preprocess,
			target_transform_b=target_transform_b,
			target_transform_type=target_transform_type
		)

	if args.dataset == "mpdd":
		return MPDDDataset(
			root=args.train_data_path,
			transform=preprocess,
			target_transform=target_transform_b,
			target_transform_type=target_transform_type,
		)

	if args.dataset == "real_iad":
		return RealIADDataset_v2(
			root=args.train_data_path,
			transform=preprocess,
			aug_rate=args.aug_rate,
			target_transform=target_transform_b,
			target_transform_type=target_transform_type
		)

	raise ValueError(f"Unsupported dataset: {args.dataset}")


def build_text_prompts(args, model, obj_list, tokenizer, device):
	with torch.amp.autocast("cuda", enabled=(device == "cuda")), torch.no_grad():
		if args.dataset == "mvtec":
			return encode_text_with_prompt_ensemble_mvtec(model, obj_list, tokenizer, device)

		if args.dataset == "visa":
			return encode_text_with_prompt_ensemble_visa(model, obj_list, tokenizer, device)

		if args.dataset == "mpdd":
			return encode_text_with_prompt_ensemble_mpdd(model, obj_list, tokenizer, device)

		if args.dataset == "real_iad":
			return encode_text_with_prompt_ensemble_real_iad(model, obj_list, tokenizer, device)

	raise ValueError(f"Unsupported dataset for text prompts: {args.dataset}")


def compute_image_level_loss(
	image_features,
	text_features,
	anomaly_labels,
	image_adaptor,
	hybrid_codebook,
	temperature,
):
	"""
	image_features:
		[B, D], frozen CLIP global image feature.

	text_features:
		[B, D, C], class 0 is normal/good, classes 1..C-1 are anomaly classes.

	anomaly_labels:
		[B], 0 for normal image, 1 for anomalous image.

	image_adaptor:
		trainable adaptor before codebook.

	hybrid_codebook:
		shared codebook. Global image feature is quantized with the same codebook.

	return:
		image_loss
		image_anomaly_score: [B]
		image_quant_loss
	"""
	image_features = image_features.float()
	text_features = text_features.float()
	anomaly_labels = anomaly_labels.float()

	adapted_image_features = image_adaptor(image_features)	# [B, D]

	global_tokens = adapted_image_features.unsqueeze(1)		# [B, 1, D]
	ret = hybrid_codebook(global_tokens)

	global_q = ret["z_q_st"].squeeze(1)						# [B, D]
	image_quant_loss = ret["quant_loss"]

	global_q = F.normalize(global_q, dim=-1)

	image_logits = torch.bmm(
		global_q.unsqueeze(1),
		text_features
	).squeeze(1) / temperature								# [B, C]

	normal_logit = image_logits[:, 0]
	anomaly_logit = torch.logsumexp(image_logits[:, 1:], dim=1)

	binary_anomaly_logit = anomaly_logit - normal_logit

	image_loss = F.binary_cross_entropy_with_logits(
		binary_anomaly_logit,
		anomaly_labels
	)

	image_anomaly_score = torch.sigmoid(binary_anomaly_logit)

	return image_loss, image_anomaly_score, image_quant_loss


def train(args):
	epochs = args.epoch
	image_size = args.image_size
	device = "cuda" if torch.cuda.is_available() else "cpu"

	os.makedirs(args.save_path, exist_ok=True)
	logger = build_logger(args.save_path)

	for arg in vars(args):
		logger.info(f"{arg}: {getattr(args, arg)}")

	with open(args.config_path, "r") as f:
		model_configs = json.load(f)

	model, _, preprocess = open_clip.create_model_and_transforms(
		args.model,
		image_size,
		pretrained=args.pretrained
	)
	model.to(device)
	model.eval()

	tokenizer = open_clip.get_tokenizer(args.model)

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

	assert args.dataset in ["mvtec", "visa", "mpdd", "real_iad"]

	train_data = build_train_dataset(
		args=args,
		preprocess=preprocess,
		target_transform_b=target_transform_b,
		target_transform_type=target_transform_type
	)

	train_dataloader = torch.utils.data.DataLoader(
		train_data,
		batch_size=args.batch_size,
		shuffle=True,
		num_workers=args.num_workers,
		pin_memory=(device == "cuda"),
		drop_last=False
	)

	trainable_layer = LinearLayer(
		model_configs["vision_cfg"]["width"],
		model_configs["embed_dim"],
		len(args.features_list),
		args.model
	).to(device)

	n_layers = len(args.features_list)
	if args.per_layer_codebook:
		# one codebook per feature layer, each reseeded and revived from its own layer
		hybrid_codebook = torch.nn.ModuleList([
			HybridCodebook(
				num_learnable=args.codebook_num_learnable,
				embed_dim=model_configs["embed_dim"],
				beta=args.beta
			) for _ in range(n_layers)
		]).to(device)
		codebooks = list(hybrid_codebook)
	else:
		hybrid_codebook = HybridCodebook(
			num_learnable=args.codebook_num_learnable,
			embed_dim=model_configs["embed_dim"],
			beta=args.beta
		).to(device)
		codebooks = [hybrid_codebook] * n_layers

	# In the stab* checkpoints the global token collapses onto one prototype
	# that no patch ever selects (tools/prototype_usage.py), so the branch only
	# occupies a codebook slot; --no_image_branch removes it altogether.
	image_adaptor = None
	image_params = []
	if not args.no_image_branch:
		image_adaptor = GlobalImageAdaptor(
			embed_dim=model_configs["embed_dim"],
			hidden_dim=args.image_adaptor_hidden_dim,
			dropout=args.image_adaptor_dropout
		).to(device)
		image_adaptor.train()
		image_params = list(image_adaptor.parameters())

	trainable_layer.train()
	hybrid_codebook.train()

	loss_focal = FocalLoss()
	loss_dice = BinaryDiceLoss()

	obj_list = train_data.get_cls_names()
	text_prompts = build_text_prompts(
		args=args,
		model=model,
		obj_list=obj_list,
		tokenizer=tokenizer,
		device=device
	)

	text_prompts = {
		k: F.normalize(v.to(device=device, dtype=torch.float32), dim=0).contiguous()
		for k, v in text_prompts.items()
	}

	print_model_stats("CLIP model", model)
	print_model_stats("Linear layer", trainable_layer)
	print_model_stats("Hybrid codebook", hybrid_codebook)
	if image_adaptor is not None:
		print_model_stats("Image adaptor", image_adaptor)

	for name, buf in hybrid_codebook.named_buffers():
		print(f"HybridCodebook BUFFER {name}: shape={tuple(buf.shape)} numel={buf.numel():,}")

	optimizer = torch.optim.Adam(
		list(trainable_layer.parameters())
		+ list(hybrid_codebook.parameters())
		+ image_params,
		lr=args.learning_rate,
		betas=(0.5, 0.999)
	)

	global_step = 0
	warmup_feats = []
	warmup_feats_layer = [[] for _ in range(n_layers)]

	def save_checkpoint(path, epoch_no):
		ckpt = {
			"epoch": epoch_no,
			"global_step": global_step,
			"trainable_linearlayer": trainable_layer.state_dict(),
			"hybrid_codebook": hybrid_codebook.state_dict(),
			"per_layer_codebook": bool(args.per_layer_codebook),
			"optimizer": optimizer.state_dict(),
		}
		if image_adaptor is not None:
			ckpt["image_adaptor"] = image_adaptor.state_dict()
		torch.save(ckpt, path)
		logger.info("Saved checkpoint to {}".format(path))

	for epoch in range(epochs):
		trainable_layer.train()
		hybrid_codebook.train()
		if image_adaptor is not None:
			image_adaptor.train()
		model.eval()

		total_loss_list = []
		construction_loss_list = []
		image_loss_list = []
		patch_quant_loss_list = []
		image_quant_loss_list = []
		image_score_list = []
		map_image_loss_list = []

		pbar = tqdm(
			train_dataloader,
			desc=f"Epoch {epoch + 1}/{epochs}",
			leave=True
		)

		for items in pbar:
			global_step += 1
			image = items["img"].to(device, non_blocking=True)
			cls_name = items["cls_name"]

			img_mask = items["img_mask"].to(device, non_blocking=True).long()
			img_mask_b = items["img_mask_b"].to(device, non_blocking=True).float()
			anomaly_label = items["anomaly"].to(device, non_blocking=True).float()

			with torch.amp.autocast("cuda", enabled=(device == "cuda")):
				with torch.no_grad():
					image_features, patch_tokens = model.encode_image(image, args.features_list)

					text_features = []
					for cls in cls_name:
						text_features.append(text_prompts[cls])
					text_features = torch.stack(text_features, dim=0)	# [B, D, C]

				if image_adaptor is not None:
					image_loss, image_anomaly_score, image_quant_loss = compute_image_level_loss(
						image_features=image_features,
						text_features=text_features,
						anomaly_labels=anomaly_label,
						image_adaptor=image_adaptor,
						hybrid_codebook=codebooks[-1],
						temperature=args.image_temperature
					)
				else:
					image_loss = image_quant_loss = torch.zeros((), device=device)
					image_anomaly_score = torch.zeros(1, device=device)

				patch_tokens_list = trainable_layer(patch_tokens)

				patch_quant_loss = 0.0
				seg_prob_list = []

				# Stabilizers. The truncated-normal init puts entries where no patch
				# lives, so which of them ever become live is seed-dependent; that is
				# a large part of the run-to-run spread. We therefore reseed the
				# codebook from real features once the adapters have warmed up, and
				# periodically revive entries that have gone unused.
				if args.codebook_init == "warmup_kmeans" and global_step <= args.codebook_warmup_steps:
					with torch.no_grad():
						for l, lt in enumerate(patch_tokens_list):
							f = lt.detach().reshape(-1, lt.shape[-1]).float()
							take = min(args.warmup_feats_per_step, f.shape[0])
							sel = torch.randperm(f.shape[0], device=f.device)[:take]
							if args.per_layer_codebook:
								warmup_feats_layer[l].append(f[sel].cpu())
							else:
								warmup_feats.append(f[sel].cpu())
					if global_step == args.codebook_warmup_steps and args.per_layer_codebook:
						for l, cb in enumerate(codebooks):
							feats = torch.cat(warmup_feats_layer[l], dim=0).to(device)
							n_init = cb.init_from_features(feats)
							logger.info(
								"layer %d: reseeded %d prototypes by spherical k-means on %d warm-up features",
								l, n_init, feats.shape[0]
							)
							warmup_feats_layer[l].clear()
							del feats
						torch.cuda.empty_cache()
					elif global_step == args.codebook_warmup_steps and len(warmup_feats) > 0:
						feats = torch.cat(warmup_feats, dim=0).to(device)
						n_init = hybrid_codebook.init_from_features(feats)
						logger.info(
							"reseeded %d prototypes by spherical k-means on %d warm-up features",
							n_init, feats.shape[0]
						)
						warmup_feats.clear()
						del feats
						torch.cuda.empty_cache()

				# init_from_features zeroes code_usage, so a revive check on the
				# reseed step itself sees every entry as dead and overwrites the
				# k-means centers with one batch of worst-covered tokens. Count
				# usage for a full interval after the reseed before judging it.
				revive_ok = (
					args.legacy_revive_at_init
					or args.codebook_init != "warmup_kmeans"
					or global_step > args.codebook_warmup_steps
				)
				if args.revive_every > 0 and global_step % args.revive_every == 0 and revive_ok:
					with torch.no_grad():
						if args.per_layer_codebook:
							n_rev = sum(
								cb.revive_dead(lt.detach().reshape(-1, lt.shape[-1]))
								for cb, lt in zip(codebooks, patch_tokens_list)
							)
						else:
							f = patch_tokens_list[-1].detach().reshape(-1, patch_tokens_list[-1].shape[-1])
							n_rev = hybrid_codebook.revive_dead(f)
					if n_rev > 0:
						logger.info("step %d: revived %d dead prototypes", global_step, n_rev)

				for layer_idx in range(len(patch_tokens_list)):
					layer_tokens = patch_tokens_list[layer_idx]

					ret = codebooks[layer_idx](layer_tokens)

					patch_tokens_q = ret["z_q_st"]
					patch_quant_loss = patch_quant_loss + ret["quant_loss"]

					patch_tokens_q = F.normalize(patch_tokens_q, dim=-1)
					seg_logits = (patch_tokens_q @ text_features) / args.temperature

					B, L, C = seg_logits.shape
					H = int(np.sqrt(L))
					if H * H != L:
						raise ValueError(f"L={L} is not square.")

					seg_logits = F.interpolate(
						seg_logits.permute(0, 2, 1).contiguous().view(B, C, H, H),
						size=image_size,
						mode="bilinear",
						align_corners=True
					)

					seg_prob = torch.softmax(seg_logits, dim=1)
					seg_prob_list.append(seg_prob)

				patch_quant_loss = patch_quant_loss / len(patch_tokens_list)

				construction_loss = 0.0

				for seg_prob_layer in seg_prob_list:
					anom_map_layer = seg_prob_layer[:, 1:, :, :].sum(dim=1)

					loss_seg = loss_focal(seg_prob_layer, img_mask)
					loss_bin = loss_dice(anom_map_layer, img_mask_b)

					construction_loss = construction_loss + 0.5 * (loss_seg + loss_bin)

				construction_loss = construction_loss / len(seg_prob_list)

				# Image-level BCE on the read-out that test.py actually reports:
				# top-k mean of the smoothed, layer-averaged pixel map. The focal and
				# dice terms weigh every normal pixel alike, so a single hot patch on
				# a normal image costs almost nothing there, yet it decides that
				# image's score. Unlike image_loss this adds no branch and no
				# parameters; it only reshapes the map the image score is read from.
				map_image_loss = torch.zeros((), device=device)
				if args.map_image_loss_weight > 0:
					# fp32 throughout: under autocast the blur (a conv) returns fp16,
					# where 1 - 1e-4 rounds to 1, the clamp stops guarding the log,
					# and a saturated map gives 0 * log(0) = NaN.
					with torch.autocast(device_type="cuda", enabled=False):
						acc_map = torch.stack(
							[p[:, 1:, :, :].float().sum(dim=1) for p in seg_prob_list], dim=0
						).mean(dim=0)
						acc_map = gaussian_blur_maps(acc_map, args.map_image_sigma)
						flat_map = acc_map.reshape(acc_map.shape[0], -1)
						k = max(1, int(round(args.map_image_topk * flat_map.shape[1])))
						map_score = torch.topk(flat_map, k, dim=1).values.mean(dim=1)
						map_score = map_score.clamp(1e-4, 1.0 - 1e-4)
						y = anomaly_label.float()
						map_image_loss = -(
							y * torch.log(map_score) + (1.0 - y) * torch.log(1.0 - map_score)
						).mean()

				loss = (
					construction_loss
					+ args.map_image_loss_weight * map_image_loss
					+ args.image_loss_weight * image_loss
					+ args.vq_weight * patch_quant_loss
					+ args.image_vq_weight * image_quant_loss
				)

			optimizer.zero_grad(set_to_none=True)
			loss.backward()

			if args.grad_clip > 0:
				torch.nn.utils.clip_grad_norm_(
					list(trainable_layer.parameters())
					+ list(hybrid_codebook.parameters())
					+ image_params,
					max_norm=args.grad_clip
				)

			optimizer.step()

			if args.save_every_steps > 0 and global_step % args.save_every_steps == 0:
				save_checkpoint(os.path.join(args.save_path, f"step_{global_step}.pth"), epoch + 1)

			total_loss_list.append(loss.item())
			construction_loss_list.append(construction_loss.item())
			image_loss_list.append(image_loss.item())
			patch_quant_loss_list.append(patch_quant_loss.item())
			image_quant_loss_list.append(image_quant_loss.item())
			image_score_list.append(image_anomaly_score.detach().mean().item())
			map_image_loss_list.append(map_image_loss.item())

			pbar.set_postfix({
				"loss": f"{loss.item():.4f}",
				"seg": f"{construction_loss.item():.4f}",
				"img": f"{image_loss.item():.4f}",
				"vq_p": f"{patch_quant_loss.item():.4f}",
				"vq_i": f"{image_quant_loss.item():.4f}",
				"img_score": f"{image_anomaly_score.detach().mean().item():.4f}",
				"map_img": f"{map_image_loss.item():.4f}",
			})

		if (epoch + 1) % args.print_freq == 0:
			logger.info(
				"epoch [{}/{}], total_loss: {:.4f}, construction_loss: {:.4f}, image_loss: {:.4f}, patch_quant_loss: {:.4f}, image_quant_loss: {:.4f}, image_score: {:.4f}".format(
					epoch + 1,
					epochs,
					np.mean(total_loss_list),
					np.mean(construction_loss_list),
					np.mean(image_loss_list),
					np.mean(patch_quant_loss_list),
					np.mean(image_quant_loss_list),
					np.mean(image_score_list),
				)
			)
			logger.info("epoch %d: map_image_loss: %.4f", epoch + 1, np.mean(map_image_loss_list))
			logger.info(
				"epoch %d: %d/%d prototypes live",
				epoch + 1, sum(cb.live_codes() for cb in set(codebooks)),
				sum(cb.num_learnable for cb in set(codebooks))
			)

		if (epoch + 1) % args.save_freq == 0:
			save_checkpoint(os.path.join(args.save_path, "epoch_" + str(epoch + 1) + ".pth"), epoch + 1)


if __name__ == "__main__":
	parser = argparse.ArgumentParser("MultiADS", add_help=True)

	# path
	parser.add_argument("--train_data_path", type=str, default="./data/mvtec", help="train dataset path")
	parser.add_argument("--save_path", type=str, default="./exps/mvtec/", help="path to save results")
	parser.add_argument("--config_path", type=str, default="./open_clip/model_configs/ViT-L-14-336.json", help="model configs")

	# model
	parser.add_argument("--dataset", type=str, default="mvtec", help="train dataset name")
	parser.add_argument("--model", type=str, default="ViT-L-14-336", help="model used")
	parser.add_argument("--pretrained", type=str, default="openai", help="pretrained weight used")
	parser.add_argument("--features_list", type=int, nargs="+", default=[6, 12, 18, 24], help="features used")
	parser.add_argument("--codebook_num_learnable", type=int, default=100, help="number of learnable entries in codebook")

	# hyper-parameter
	parser.add_argument("--epoch", type=int, default=10, help="epochs")
	parser.add_argument("--learning_rate", type=float, default=0.001, help="learning rate")
	parser.add_argument("--batch_size", type=int, default=4, help="batch size")
	parser.add_argument("--temperature", type=float, default=0.01, help="temperature for pixel-level contrastive segmentation")
	parser.add_argument("--image_temperature", type=float, default=0.1, help="temperature for global image-level anomaly score")
	parser.add_argument("--image_loss_weight", type=float, default=0.0, help="weight for image-level anomaly score loss; 0 is the reported objective (the image-level BCE hurt localization), set 0.03 to reproduce the earlier sweeps")
	parser.add_argument("--image_vq_weight", type=float, default=0.1, help="weight for image-level codebook quantization loss")
	parser.add_argument("--map_image_loss_weight", type=float, default=0.0, help="weight for an image-level BCE on the top-k mean of the smoothed pixel map, i.e. on the read-out test.py reports; 0 disables")
	parser.add_argument("--map_image_topk", type=float, default=0.001, help="rho of the top-k read-out inside the map image loss")
	parser.add_argument("--map_image_sigma", type=float, default=4.0, help="Gaussian sigma applied to the map before the top-k inside the map image loss")
	parser.add_argument("--image_adaptor_hidden_dim", type=int, default=None, help="hidden dim of image adaptor, None means embed_dim")
	parser.add_argument("--image_adaptor_dropout", type=float, default=0.1, help="dropout for image adaptor")
	parser.add_argument("--no_image_branch", action="store_true", help="drop the global-token branch (image adaptor, its quantization loss and the image BCE); the codebook then sees patch tokens only")

	parser.add_argument("--image_size", type=int, default=518, help="image size")
	parser.add_argument("--aug_rate", type=float, default=0.2, help="augmentation rate")
	parser.add_argument("--print_freq", type=int, default=1, help="print frequency")
	parser.add_argument("--save_freq", type=int, default=1, help="save frequency")
	parser.add_argument("--save_every_steps", type=int, default=0, help="also save step_<N>.pth every N optimizer steps; 0 disables")
	parser.add_argument("--seed", type=int, default=42, help="random seed")
	parser.add_argument("--num_workers", type=int, default=4, help="dataloader workers")
	parser.add_argument("--grad_clip", type=float, default=1.0, help="gradient clipping max norm, <=0 disables clipping")

	# codebook
	parser.add_argument("--codebook_init", type=str, default="randn", choices=["randn", "warmup_kmeans"], help="how the learnable entries start: truncated normal, or spherical k-means over features collected during a short warm-up")
	parser.add_argument("--codebook_warmup_steps", type=int, default=100, help="steps of warm-up before the k-means reseed")
	parser.add_argument("--warmup_feats_per_step", type=int, default=256, help="patch features sampled per layer per warm-up step")
	parser.add_argument("--revive_every", type=int, default=0, help="every N steps, reseed prototypes unused since the last check with poorly covered features; 0 disables")
	parser.add_argument("--legacy_revive_at_init", action="store_true", help="reproduce the stab* checkpoints: allow a revive check on the k-means reseed step, which then replaces every center (usage was just zeroed) with one batch of worst-covered layer-24 tokens")
	parser.add_argument("--beta", type=float, default=0.1, help="commitment loss weight inside codebook")
	parser.add_argument("--vq_weight", type=float, default=0.25, help="weight for patch-level quantization loss")
	parser.add_argument("--per_layer_codebook", action="store_true", help="one codebook of codebook_num_learnable entries per feature layer instead of one shared by all layers")

	args = parser.parse_args()
	if args.no_image_branch and args.image_loss_weight > 0:
		parser.error("--image_loss_weight acts on the global-token branch, which --no_image_branch removes")

	setup_seed(args.seed)
	train(args)