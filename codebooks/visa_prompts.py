from typing import Union, List
from pkg_resources import packaging
import torch
import numpy as np
import torch.nn.functional as F
import pdb

# Dictionary of MVTec categories and their specific defect types
VISA_DEFECT_MAP = {
    'candle': ['normal', 'damage', 'weird wick', 'partical', 'melded', 'spot', 'extra', 'missing'],
    'capsules': ['normal', 'scratch', 'bubble', 'discolor', 'leak'],
    'cashew': ['normal', 'scratch', 'breakage','burnt', 'stuck', 'hole', 'spot'],
    'chewinggum': ['normal', 'scratch', 'spot', 'missing'],
    'fryum': ['normal', 'scratch', 'breakage', 'burnt', 'stuck', 'spot'],
    'macaroni1': ['normal', 'scratch', 'crack', 'spot', 'chip'],
    'macaroni2': ['normal', 'scratch', 'breakage', 'crack', 'spot', 'chip'],
    'pcb1': ['normal', 'scratch', 'melt', 'bent', 'missing'],
    'pcb2': ['normal', 'scratch', 'melt', 'bent', 'missing'],
    'pcb3': ['normal', 'scratch', 'melt', 'bent', 'missing'],
    'pcb4': ['normal', 'scratch', 'damage', 'extra', 'burnt', 'missing', 'wrong place'],
    'pipe_fryum': ['normal', 'scratch', 'breakage', 'burnt', 'stuck', 'spot']
}

normal = [
    '{}', 'flawless {}', 'perfect {}', 'unblemished {}', '{} without flaw', 
    '{} without defect', '{} without damage', '{} in pristine condition', 
    '{} with no imperfections', '{} with ideal quality'
]

damage = [
    '{} has a damaged defect', 'flawed {} with damage', '{} shows signs of damage', 
    'damage found on {}', '{} with visible wear and tear', '{} with structural damage'
]

scratch = [
    '{} has a scratch defect', 'flawed {} with a scratch', 'scratches visible on {}', 
    '{} has surface scratches', 'small scratches found on {}', '{} with scratch marks'
]

breakage = [
    '{} with a breakage defect', 'broken {}', '{} with broken defect', 
    '{} shows breakage', 'broken or cracked areas on {}', 'visible breakage on {}'
]

burnt = [
    '{} with a burnt defect', '{} shows burn marks', 'burnt areas on {}', 
    '{} with signs of burning', 'scorch marks on {}', '{} appears slightly burnt'
]

weird_wick = [
    '{} with a weird wick defect', '{} has an unusual wick', 'the wick on {} appears odd', 
    '{} with a strangely shaped wick', 'irregular wick found on {}', 'odd wick defect on {}'
]

stuck = [
    '{} with a stuck defect', '{} stuck together', '{} has stuck parts', 
    'adhesive issue causing {} to stick', '{} is partially stuck', '{} with adhesion defect'
]

crack = [
    '{} with a crack defect', '{} has a visible crack', 'cracked areas on {}', 
    '{} with surface cracking', 'fine cracks found on {}', '{} shows crack lines'
]

wrong_place = [
    '{} with defect that something on wrong place', '{} has a misplaced defect', 
    'flawed {} with misplacing', 'misaligned part on {}', '{} shows parts out of place', 
    'misplacement detected on {}'
]

partical = [
    '{} with particles defect', '{} has foreign particles', 'small particles on {}', 
    '{} with unwanted particles', 'contaminants found on {}', '{} with visible particles'
]

bubble = [
    '{} with bubbles defect', 'bubbles seen on {}', '{} with bubble marks', 
    'air bubbles in {}', '{} contains bubble defects', 'small bubbles on {} surface'
]

melded = [
    '{} with melded defect', 'melded parts on {}', '{} has fused areas', 
    'fused spots on {}', 'melded areas on {}', '{} with melded material'
]

hole = [
    '{} has a hole defect', 'a hole on {}', 'visible hole on {}', 
    '{} has small punctures', '{} shows perforations', 'hole present on {}'
]

melt = [
    '{} with melt defect', 'melted areas on {}', '{} shows melting', 
    'signs of melting on {}', '{} with melted spots', '{} has a melted appearance'
]

bent = [
    '{} has a bent defect', 'flawed {} with a bent', 'bent areas on {}', 
    '{} with visible bending', '{} shows curvature issues', 'noticeable bend in {}'
]

spot = [
    '{} with spot defect', 'spots visible on {}', 'flawed {} with spots', 
    '{} with visible spotting', '{} shows small spots', 'surface spots on {}'
]

extra = [
    '{} with extra thing', '{} has a defect with extra thing', 'extra material on {}', 
    '{} contains additional pieces', '{} with extra component defect', 'unwanted additions on {}'
]

chip = [
    '{} with chip defect', '{} with fragment broken defect', 'chipped areas on {}', 
    '{} with chipped parts', 'broken fragments on {}', 'chip marks found on {}'
]

missing = [
    '{} with a missing defect', 'flawed {} with something missing', '{} has missing parts', 
    'missing components on {}', 'absent pieces in {}', '{} is incomplete'
]
discolor = [
    '{} has a color defect', 'inconsistent color on {}', '{} with color discrepancies', 
    '{} has a noticeable color difference', '{} with irregular coloring', 
    '{} has off-color patches'
]
leak = [
    '{} has a leak defect', '{} with oil', 'flawed {} with liquid', '{} with leaking issue'
]

prompt_state_dict = {
  'xxx': xxx
}

def generate_semantic_prompts(obj_name):
    """Generates an ensemble of prompts for each defect type."""
    if obj_name not in MVTEC_DEFECT_MAP:
        return []
    
    prompt_templates = ['a bad photo of a {}.','a low resolution photo of the {}.', 'a bad photo of the {}.', 'a cropped photo of the {}.', 'a bright photo of a {}.', 'a dark photo of the {}.', 'a photo of my {}.', 'a photo of the cool {}.', 'a close-up photo of a {}.', 'a black and white photo of the {}.', 'a bright photo of the {}.', 'a cropped photo of a {}.', 'a jpeg corrupted photo of a {}.', 'a blurry photo of the {}.', 'a photo of the {}.', 'a good photo of the {}.', 'a photo of one {}.', 'a close-up photo of the {}.', 'a photo of a {}.', 'a low resolution photo of a {}.', 'a photo of a large {}.', 'a blurry photo of a {}.', 'a jpeg corrupted photo of the {}.', 'a good photo of a {}.', 'a photo of the small {}.', 'a photo of the large {}.', 'a black and white photo of a {}.', 'a dark photo of a {}.', 'a photo of a cool {}.', 'a photo of a small {}.', 'there is a {} in the scene.', 'there is the {} in the scene.', 'this is a {} in the scene.', 'this is the {} in the scene.', 'this is one {} in the scene.']

    
    obj_prompts = []
    for defect in MVTEC_DEFECT_MAP[obj_name]:
        for defect_prompt in prompt_state_dict[defect]:
            for t in prompt_templates:
                obj_prompts.append(t.format(defect_prompt.format(obj_name)))

    return obj_prompts # List of Lists: [[defect1_v1, defect1_v2...], [defect2_v1...]]


def get_semantic_codes(model, objs, tokenizer, device):
    """
    Args:
        model: loaded CLIP model
        objs: obj list 
        tokenizer: CLIP's tokenizer
        device: 'cuda' or 'cpu'
    Returns:
        semantic_codes: shape [num_defects, embed_dim]
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