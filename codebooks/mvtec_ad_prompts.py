from typing import Union, List
from pkg_resources import packaging
import torch
import numpy as np
import torch.nn.functional as F
import pdb

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

good = [
    '{}', 'flawless {}', 'perfect {}', 'unblemished {}', '{} without flaw', 
    '{} without defect', '{} without damage', '{} in ideal condition', 
    '{} with immaculate quality', '{} without any imperfections'
]

bent = [
    '{} has a bent defect', 'flawed {} with a bent', 'a bend found in {}', 
    '{} with noticeable bending', 'a bent edge on {}', '{} has a slight curve defect'
]

broken = [
    '{} has a broken defect', 'flawed {} with breakage', 'visible breakage on {}', 
    '{} with broken areas', '{} shows signs of breaking', 'cracked or broken spots on {}'
]

color = [
    '{} has a color defect', 'inconsistent color on {}', '{} with color discrepancies', 
    '{} has a noticeable color difference', '{} with irregular coloring', 
    '{} has off-color patches'
]

combined = [
    '{} has a combined defect', 'multiple issues with {}', '{} with mixed defects', 
    '{} showing multiple imperfections', 'several flaws found on {}', 
    '{} with combined defect types'
]

contamination = [
    '{} has a contamination defect', 'foreign particles on {}', 
    '{} is contaminated', '{} contains contaminants', '{} has impurity issues', 
    'traces of contamination on {}'
]

crack = [
    '{} has a crack defect', 'a crack is present on {}', 'cracked area on {}', 
    '{} with noticeable cracking', 'fine cracks found on {}', 
    '{} shows surface cracks'
]

cut = [
    '{} has a cut defect', 'cut marks on {}', '{} with visible cuts', 
    'a cut detected on {}', '{} is sliced or cut', 'surface cut seen on {}'
]

fabric = [
    '{} has a fabric defect', '{} has a fabric border defect', '{} has a fabric interior defect', 
    'fabric quality issues on {}', '{} with textile irregularities', 
    'fabric borders on {} show defects'
]

faulty_imprint = [
    '{} has a faulty imprint defect', '{} has a print defect', 'incorrect printing on {}', 
    'misaligned print on {}', 'printing errors present on {}', '{} has a blurred print defect'
]

glue = [
    '{} has a glue defect', '{} has a glue strip defect', 'excess glue on {}', 
    '{} with uneven glue application', '{} has visible glue spots', 
    'misplaced glue seen on {}'
]

hole = [
    '{} has a hole defect', 'a hole on {}', 'visible hole on {}', 
    '{} with punctures', 'small hole found in {}', 'perforations present on {}'
]

missing = [
    '{} has a missing defect', 'flawed {} with something missing', '{} has missing components', 
    'missing parts on {}', '{} shows absent pieces', 'certain parts missing from {}'
]

poke = [
    '{} has a poke defect', '{} has a poke insulation defect', 'visible poke mark on {}', 
    '{} has puncture marks', 'a poke flaw on {}', 'small poke defect on {}'
]

rough = [
    '{} has a rough defect', 'rough texture on {}', 'uneven surface on {}', 
    '{} is coarser than expected', 'surface roughness seen on {}', 
    'texture defects on {}'
]

scratch = [
    '{} has a scratch defect', 'flawed {} with a scratch', 'visible scratches on {}', 
    '{} with surface scratches', 'minor scratches seen on {}', '{} shows scratch marks'
]

squeeze = [
    '{} has a squeeze defect', 'flawed {} with a squeeze', 'squeezed area on {}', 
    '{} has compression marks', '{} appears squeezed', 'flattened areas on {}'
]

thread = [
    '{} has a thread defect', 'flawed {} with a thread', 'loose threads on {}', 
    '{} has visible threads', 'untrimmed threads on {}', 'threads sticking out on {}'
]

liquid = [
    '{} has a liquid defect', 'flawed {} with liquid', '{} with oil', 
    'liquid marks on {}', '{} with liquid residue', 'stains from liquid on {}'
]

misplaced = [
    '{} has a misplaced defect', 'flawed {} with misplacing', '{} shows misalignment', 
    'misplaced parts on {}', '{} with incorrect positioning', 'positioning defects on {}'
]

damaged = [
    '{} has a damaged defect', 'flawed {} with damage', '{} with visible damage', 
    'damaged areas on {}', 'physical damage seen on {}', 'noticeable wear on {}'
]

prompt_state_dict = {
    'good': good,
    'bent': bent,
    'broken': broken,
    'color': color,
    'combined': combined,
    'contamination': contamination,
    'crack': crack,
    'cut': cut,
    'fabric': fabric,
    'faulty_imprint' : faulty_imprint,
    'glue': glue,
    'hole': hole,
    'missing': missing,
    'poke': poke,
    'rough': rough,
    'scratch': scratch,
    'squeeze': squeeze,
    'thread': thread,
    'liquid': liquid,
    'misplaced': misplaced,
    'damaged':damaged

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