# Copyright (c) 2025 Robert Bosch GmbH
# SPDX-License-Identifier: AGPL-3.0

from torch import Tensor, nn
import torch
from torch.nn import functional as F
import pdb

class LinearLayer(nn.Module):
	def __init__(self, dim_in, dim_out, k, model):
		super(LinearLayer, self).__init__()
		if 'ViT' in model:
			self.fc = nn.ModuleList([nn.Linear(dim_in, dim_out) for i in range(k)])
		else:
			self.fc = nn.ModuleList([nn.Linear(dim_in * 2 ** (i + 2), dim_out) for i in range(k)])

	def forward(self, tokens):
		for i in range(len(tokens)):
			if len(tokens[i].shape) == 3:
				tokens[i] = self.fc[i](tokens[i][:, 1:, :])
			else:
				B, C, H, W = tokens[i].shape
				tokens[i] = self.fc[i](tokens[i].view(B, C, -1).permute(0, 2, 1).contiguous())
		return tokens


import torch
import torch.nn as nn
import torch.nn.functional as F
	

class HybridCodebook(nn.Module):
	def __init__(self, semantic_embeddings=None, num_learnable=128, embed_dim=1024, beta=0.25,
				 calib_semantic_embeddings=None):
		super().__init__()
		if semantic_embeddings is not None:
			self.register_buffer(
				"frozen_semantic_entries",
				F.normalize(semantic_embeddings, dim=-1)
			)
			self.num_semantic = self.frozen_semantic_entries.size(0)
		else:
			self.frozen_semantic_entries = None
			self.num_semantic = 0

		# Calibration-path semantics. These target-vocabulary entries are NOT part
		# of the codebook: they never enter get_full_codebook, never win an argmax
		# and never become z_q. They participate only in the assignment
		# distribution whose entropy drives the per-pixel temperature, so the
		# coverage question becomes "is this patch explained by a learned
		# prototype OR by a concept the operator named?" while the quantized value
		# stays an image-manifold vector. This sidesteps both failure modes of
		# appending them to the codebook proper: the modality gap cannot stop them
		# contributing probability mass, and they cannot corrupt the text-anchor
		# scoring by substituting a text vector into an image-vs-text comparison.
		if calib_semantic_embeddings is not None:
			self.register_buffer(
				"calib_semantic_entries",
				F.normalize(calib_semantic_embeddings, dim=-1)
			)
			self.num_calib_semantic = self.calib_semantic_entries.size(0)
		else:
			self.calib_semantic_entries = None
			self.num_calib_semantic = 0

		self.num_learnable = num_learnable
		self.embed_dim = embed_dim
		self.beta = beta

		self.learnable_entries = nn.Parameter(torch.randn(num_learnable, embed_dim))
		nn.init.trunc_normal_(self.learnable_entries, std=0.02)

		# How often each learnable entry has been selected since the last reset.
		# Entries that are never selected receive no gradient from either term of
		# L_quant, so with a random init the set of live codes is itself a random
		# variable: a large part of the run-to-run spread comes from here.
		self.register_buffer("code_usage", torch.zeros(num_learnable, dtype=torch.long))

	@torch.no_grad()
	def init_from_features(self, feats, iters=10):
		"""Spherical k-means init of the learnable entries from real features.

		feats: [N, D], not necessarily normalized. Seeds with k-means++ on cosine
		distance, then runs Lloyd iterations on the unit sphere. Replaces the
		truncated-normal init, which places entries where no patch lives and
		leaves their fate to chance.
		"""
		feats = F.normalize(feats.float(), dim=-1)
		n, _ = feats.shape
		k = self.num_learnable
		if n < k:
			return 0

		start = torch.randint(0, n, (1,), device=feats.device).item()
		centers = feats[start: start + 1].clone()
		d = (1.0 - feats @ centers.t()).squeeze(1).clamp_min(0)
		for _ in range(k - 1):
			# Once every remaining feature coincides with a chosen center the
			# distances are all zero and multinomial has no distribution to draw
			# from; fall back to a uniform pick so seeding still terminates.
			total = d.sum()
			probs = d / total if total > 0 else torch.full_like(d, 1.0 / n)
			nxt = torch.multinomial(probs, 1).item()
			c = feats[nxt: nxt + 1]
			centers = torch.cat([centers, c], dim=0)
			d = torch.minimum(d, (1.0 - feats @ c.t()).squeeze(1).clamp_min(0))

		for _ in range(iters):
			assign = (feats @ centers.t()).argmax(dim=1)
			for j in range(k):
				m = assign == j
				if m.any():
					centers[j] = F.normalize(feats[m].mean(dim=0), dim=-1)

		self.learnable_entries.data.copy_(centers)
		self.code_usage.zero_()
		return k

	@torch.no_grad()
	def revive_dead(self, feats, min_count=1):
		"""Re-seed entries unused since the last reset with poorly covered features.

		Returns the number revived. Keeps the effective codebook size from drifting
		with the seed, which is what makes K comparable across runs.
		"""
		dead = (self.code_usage < min_count).nonzero(as_tuple=False).squeeze(1)
		self.code_usage.zero_()
		if dead.numel() == 0:
			return 0

		feats = F.normalize(feats.float(), dim=-1)
		live = F.normalize(self.learnable_entries.data, dim=-1)
		worst = (feats @ live.t()).max(dim=1).values.argsort()[: dead.numel()]
		self.learnable_entries.data[dead] = feats[worst]
		return int(dead.numel())

	def live_codes(self):
		return int((self.code_usage > 0).sum().item())

	def get_full_codebook(self):
		learned_norm = F.normalize(self.learnable_entries, dim=-1)
		if self.frozen_semantic_entries is not None:
			return torch.cat([self.frozen_semantic_entries, learned_norm], dim=0)
		else:
			return learned_norm

	def forward(self, x, assign_temperature=1.0, semantic_bias=0.0, assign_mode="raw"):
		x = F.normalize(x, dim=-1)

		codebook = self.get_full_codebook()					# [K, C]
		raw_logits = torch.matmul(x, codebook.t())			# [B, L, K]
		logits = raw_logits

		# The two halves are not on comparable footing: learnable prototypes are
		# fitted inside the cloud of adapted patch features, whereas the semantic
		# entries sit in the text region of the joint space, so on raw cosine the
		# semantic half is almost never the argmax. Three ways to ask what those
		# entries would contribute if the comparison were fair:
		#   bias    a constant offset on the semantic logits (crude, no structure)
		#   center  remove each half's mean direction, which is what the modality
		#           gap is: a constant offset between the image and text cones
		#   zscore  standardize the similarities within each half, so the halves
		#           compete on rank rather than on absolute cosine
		if self.num_semantic > 0 and assign_mode == "center":
			sem = codebook[:self.num_semantic]
			lrn = codebook[self.num_semantic:]
			mu_sem = F.normalize(sem.mean(dim=0, keepdim=True), dim=-1)
			mu_lrn = F.normalize(lrn.mean(dim=0, keepdim=True), dim=-1)
			x_c = F.normalize(x - mu_lrn, dim=-1)			# patches live in the image cone
			cb_c = torch.cat([
				F.normalize(sem - mu_sem, dim=-1),
				F.normalize(lrn - mu_lrn, dim=-1),
			], dim=0)
			logits = torch.matmul(x_c, cb_c.t())
		elif self.num_semantic > 0 and assign_mode == "zscore":
			sem_l = raw_logits[..., :self.num_semantic]
			lrn_l = raw_logits[..., self.num_semantic:]
			sem_z = (sem_l - sem_l.mean(dim=-1, keepdim=True)) / (sem_l.std(dim=-1, keepdim=True) + 1e-6)
			lrn_z = (lrn_l - lrn_l.mean(dim=-1, keepdim=True)) / (lrn_l.std(dim=-1, keepdim=True) + 1e-6)
			logits = torch.cat([sem_z, lrn_z], dim=-1)

		if semantic_bias != 0.0 and self.num_semantic > 0:
			bias = torch.zeros(logits.size(-1), device=logits.device, dtype=logits.dtype)
			bias[:self.num_semantic] = semantic_bias
			logits = logits + bias

		indices = torch.argmax(logits, dim=-1)				# [B, L]
		z_q = codebook[indices]								# [B, L, C]
		z_q_st = x + (z_q - x).detach()

		is_learnable = (indices >= self.num_semantic).float()	# [B, L]
		num_selected = is_learnable.sum()

		if self.training:
			sel = indices[indices >= self.num_semantic] - self.num_semantic
			if sel.numel() > 0:
				self.code_usage.index_add_(
					0, sel.reshape(-1), torch.ones_like(sel.reshape(-1))
				)

		# codebook loss: only for learnable entries
		pos_sim_codebook = F.cosine_similarity(x.detach(), z_q, dim=-1)
		per_token_codebook_loss = 1.0 - pos_sim_codebook
		vq_loss = (per_token_codebook_loss * is_learnable).sum() / (num_selected + 1e-6)

		# commitment loss: only for learnable entries
		pos_sim_commit = F.cosine_similarity(x, z_q.detach(), dim=-1)
		per_token_commitment_loss = 1.0 - pos_sim_commit
		commitment_loss = (per_token_commitment_loss * is_learnable).sum() / (num_selected + 1e-6)

		quant_loss = vq_loss + self.beta * commitment_loss

		# Calibration is defined on the raw cosine distribution (Eq. for U_i), so it
		# must not move when assign_mode or semantic_bias changes: those alter which
		# entry wins, not how well the codebook covers the patch. Deriving
		# assign_probs from the adjusted logits would confound the two.
		calib_logits = raw_logits
		resid_indices = None
		resid_logits = None
		if self.calib_semantic_entries is not None:
			sem_logits = torch.matmul(x, self.calib_semantic_entries.t())
			# Union distribution: assignment is unchanged, coverage is not.
			calib_logits = torch.cat([raw_logits, sem_logits], dim=-1)

			# Residual quantization diagnostic. What the winning prototype failed
			# to explain is re-quantized against the named concepts alone, so the
			# text entries compete only with each other and absolute cosine scale
			# stops mattering. z_q is unaffected; this is a read-out, not a path.
			resid = F.normalize(x - z_q, dim=-1)
			resid_logits = torch.matmul(resid, self.calib_semantic_entries.t())
			resid_indices = torch.argmax(resid_logits, dim=-1)

		assign_probs = torch.softmax(calib_logits / assign_temperature, dim=-1)

		return {
			"logits": logits,
			"raw_logits": raw_logits,
			"assign_probs": assign_probs,
			"resid_indices": resid_indices,
			"resid_logits": resid_logits,
			"indices": indices,
			"z_q": z_q,
			"z_q_st": z_q_st,
			"vq_loss": vq_loss,
			"commitment_loss": commitment_loss,
			"quant_loss": quant_loss,
			"is_learnable": is_learnable,
		}