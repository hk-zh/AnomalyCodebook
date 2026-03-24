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
	def __init__(self, semantic_embeddings, num_learnable=128, embed_dim=1024, beta=0.25):
		super().__init__()

		self.register_buffer(
			"frozen_semantic_entries",
			F.normalize(semantic_embeddings, dim=-1)
		)
		self.num_semantic = self.frozen_semantic_entries.size(0)
		self.num_learnable = num_learnable
		self.embed_dim = embed_dim
		self.beta = beta

		self.learnable_entries = nn.Parameter(torch.randn(num_learnable, embed_dim))
		nn.init.trunc_normal_(self.learnable_entries, std=0.02)

	def get_full_codebook(self):
		learned_norm = F.normalize(self.learnable_entries, dim=-1)
		return torch.cat([self.frozen_semantic_entries, learned_norm], dim=0)

	def forward(self, x):
		x = F.normalize(x, dim=-1)

		codebook = self.get_full_codebook()					# [K, C]
		logits = torch.matmul(x, codebook.t())				# [B, L, K]

		indices = torch.argmax(logits, dim=-1)				# [B, L]
		z_q = codebook[indices]								# [B, L, C]

		z_q_st = x + (z_q - x).detach()

		is_learnable = (indices >= self.num_semantic).float()	# [B, L]

		# commitment: update x / encoder side
		pos_sim_commit = F.cosine_similarity(x, z_q.detach(), dim=-1)   # [B, L]
		commitment_loss = (1.0 - pos_sim_commit).mean()

		# codebook: update learnable entries only
		pos_sim_codebook = F.cosine_similarity(x.detach(), z_q, dim=-1) # [B, L]
		per_token_codebook_loss = 1.0 - pos_sim_codebook
		vq_loss = (per_token_codebook_loss * is_learnable).sum() / (is_learnable.sum() + 1e-6)

		quant_loss = vq_loss + self.beta * commitment_loss

		return {
			"logits": logits,
			"indices": indices,
			"z_q": z_q,
			"z_q_st": z_q_st,
			"vq_loss": vq_loss,
			"commitment_loss": commitment_loss,
			"quant_loss": quant_loss,
		}