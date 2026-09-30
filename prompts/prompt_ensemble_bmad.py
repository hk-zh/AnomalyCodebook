# Copyright (c) 2025 Robert Bosch GmbH
# SPDX-License-Identifier: AGPL-3.0
#
# BMAD medical anomaly-segmentation prompts. Binary (normal vs lesion).
# The raw class names ("brain", "resc", "his") are not descriptive to CLIP,
# so we map each to a modality-aware phrase (e.g. "brain MRI", "retinal OCT").

import torch

# class name (from meta.json) -> descriptive medical phrase for CLIP
CLS2DESC = {
    "brain": "brain MRI",
    "liver": "liver CT scan",
    "resc": "retinal OCT",
    "oct2017": "retinal OCT",
    "chest": "chest X-ray",
    "his": "histopathology image",
}


def encode_text_with_prompt_ensemble(model, objs, tokenizer, device):
    good = [
        "{}", "normal {}", "healthy {}", "{} with no lesion",
        "{} without abnormality", "{} in normal condition",
        "{} showing no disease", "unremarkable {}",
    ]
    lesion = [
        "{} with a lesion", "{} with an abnormality", "abnormal {}",
        "{} with a tumor", "{} showing a pathological region",
        "diseased {}", "{} with a lesion region", "{} with an anomalous region",
    ]

    prompt_state = [good, lesion]

    prompt_templates = [
        "a photo of a {}.", "a medical image of a {}.", "a scan showing a {}.",
        "a close-up medical scan of a {}.", "a grayscale medical image of a {}.",
        "a diagnostic image of a {}.", "this is a {}.", "a cropped medical image of a {}.",
    ]

    text_prompts = {}
    for obj in objs:
        desc = CLS2DESC.get(obj, obj)
        text_features = []
        for state_list in prompt_state:
            prompted_state = [s.format(desc) for s in state_list]
            prompted_sentence = []
            for s in prompted_state:
                for template in prompt_templates:
                    prompted_sentence.append(template.format(s))
            prompted_sentence = tokenizer(prompted_sentence).to(device)
            class_embeddings = model.encode_text(prompted_sentence)
            class_embeddings /= class_embeddings.norm(dim=-1, keepdim=True)
            class_embedding = class_embeddings.mean(dim=0)
            class_embedding /= class_embedding.norm()
            text_features.append(class_embedding)
        text_features = torch.stack(text_features, dim=1).to(device)
        text_prompts[obj] = text_features

    return text_prompts
