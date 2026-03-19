from typing import Union, List
from pkg_resources import packaging
import torch
import numpy as np
import torch.nn.functional as F

# Dictionary of MVTec categories and their specific defect types
MVTEC_DEFECT_MAP = {
    "bottle": ["broken large", "broken small", "contamination"],
    "cable": ["bent", "cable swap", "cut inner insulation", "cut outer insulation", "missing cable", "missing wire", "poke insulation"],
    "capsule": ["faulty imprint", "poke", "scratch", "squeeze"],
    "carpet": ["color", "cut", "hole", "metal contamination", "thread"],
    "grid": ["bent", "broken", "glue", "metal contamination", "thread"],
    "hazelnut": ["crack", "cut", "hole", "print"],
    "leather": ["color", "cut", "fold", "glue", "poke"],
    "metal_nut": ["bent", "color", "flip", "scratch"],
    "pill": ["color", "combined", "contamination", "crack", "faulty imprint", "pill type", "scratch"],
    "screw": ["manipulated front", "scratch head", "scratch neck", "thread side", "thread top"],
    "tile": ["crack", "glue", "oil", "rough", "scratch"],
    "toothbrush": ["defective"],
    "transistor": ["bent lead", "cut lead", "damaged case", "misplaced"],
    "wood": ["combined", "hole", "liquid", "scratch"],
    "zipper": ["broken teeth", "combined", "fabric border", "fabric interior", "rough", "split teeth"]
}

def generate_semantic_prompts(obj_name):
    """Generates an ensemble of prompts for each defect type."""
    if obj_name not in MVTEC_DEFECT_MAP:
        return []
    
    templates = [
        "a cropped image patch of {} on {}",
        "a close-up photo of {} texture",
        "an image area showing {}",
        "a patch of {} with {}"
    ]
    
    obj_prompts = []
    for defect in MVTEC_DEFECT_MAP[obj_name]:
        # For each defect, create an ensemble of 4 sentences
        defect_ensemble = [t.format(defect, obj_name) if "{}" in t and t.count("{}") == 2 
                           else t.format(defect) for t in templates]
        obj_prompts.append(defect_ensemble)
        
    return obj_prompts # List of Lists: [[defect1_v1, defect1_v2...], [defect2_v1...]]


def get_semantic_codes(model, objs, tokenizer, device):
    """
    Args:
        model: 加载好的 CLIP 模型 (例如 OpenCLIP)
        objs: 缺陷名称列表 (例如 ['crack', 'scratch', 'hole'])
        tokenizer: CLIP 对应的 tokenizer
        device: 'cuda' 或 'cpu'
    Returns:
        semantic_codes: shape [num_defects, embed_dim] 的张量
    """
    
    semantic_codes = []
    
    model.eval()
    with torch.no_grad():
        for defect in objs:
            prompts = generate_semantic_prompts(defect)
            
            tokens = tokenizer(prompts).to(device)
            embeddings = model.encode_text(tokens) # [num_templates, embed_dim]
            embeddings = F.normalize(embeddings, dim=-1)
        
            semantic_codes.append(embeddings)
            
    return torch.cat(semantic_codes, dim=0)