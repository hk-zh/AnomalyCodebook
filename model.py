# Copyright (c) 2025 Robert Bosch GmbH
# SPDX-License-Identifier: AGPL-3.0

from torch import Tensor, nn
import torch
from torch.nn import functional as F

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


class HybridCodebook(nn.Module):
    def __init__(self, semantic_embeddings, num_learnable=128, embed_dim=768):
        """
        Args:
            semantic_embeddings: Tensor of shape [num_classes, embed_dim] 
                                derived from CLIP text encoder.
            num_learnable: Number of entries the model can learn for 'normal' patterns.
            embed_dim: The dimensionality of the CLIP latent space.
        """
        super().__init__()
        
        # 1. Frozen Semantic Part
        # register_buffer ensures these are saved in the checkpoint but NOT updated by the optimizer.
        # We normalize them immediately to place them on the CLIP hypersphere.
        self.register_buffer(
            "frozen_semantic_entries", 
            F.normalize(semantic_embeddings, dim=-1)
        )
        self.num_semantic = self.frozen_semantic_entries.size(0)
        
        # 2. Learnable Part
        # These are standard parameters that will be updated during training.
        self.learnable_entries = nn.Parameter(torch.randn(num_learnable, embed_dim))
        
        # Proper initialization is key for learnable codebooks
        nn.init.trunc_normal_(self.learnable_entries, std=0.02)

    def get_full_codebook(self):
        """
        Returns the combined codebook where the first section is frozen text
        and the second is the current state of learnable tokens.
        """
        # We normalize the learnable entries before concatenation so they 
        # exist in the same vector space/scale as the CLIP embeddings.
        learned_norm = F.normalize(self.learnable_entries, dim=-1)
        
        return torch.cat([self.frozen_semantic_entries, learned_norm], dim=0)

def forward(self, x):
        """
        Args:
            x: projected patch features [B, L, C]
        Returns:
            logits: similarity scores [B, L, Total_Entries]
            selected_codes: the actual embeddings selected from the codebook [B, L, C]
        """
        # 1. Get the combined codebook (Frozen Semantics + Learnable Normals)
        # Shape: [num_semantic + num_learnable, C]
        codebook = self.get_full_codebook()
        
        # 2. Similarity calculation (dot product)
        # Result shape: [B, L, Total_Entries]
        logits = torch.matmul(x, codebook.t())
        
        # 3. Selection (Quantization)
        # Find the index of the most similar codebook entry for each patch
        # Shape: [B, L]
        selected_indices = torch.argmax(logits, dim=-1)
        
        # 4. Retrieve the selected embeddings
        # Shape: [B, L, C]
        selected_codes = codebook[selected_indices]
        
        return logits, selected_codes