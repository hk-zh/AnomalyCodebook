# Copyright (c) 2025 Robert Bosch GmbH
# SPDX-License-Identifier: AGPL-3.0
#
# GoodsAD prompts. Retail products (cigarette boxes, drink bottles and cans,
# food boxes and packages) whose defect states are: broken, cap_half_open,
# cap_open, deformation, opened, straw_missing, surface_anomaly,
# surface_damage. The class space below covers those states.

import torch

# Unified global class space for GoodsAD
GLOBAL_DEFECT_CLASSES = [
	"good",
	"opened",
	"cap open",
	"broken",
	"deformed",
	"missing",
	"surface damage",
	"surface anomaly",
]


def _get_anchor_prompt_bank():
	"""
	Object-agnostic class prompts.
	All objects share the same global class space.
	"""

	return {
		"good": [
			"a normal region",
			"a defect-free region",
			"an intact region",
			"a properly sealed region",
		],

		"opened": [
			"an opened package region",
			"a region with a torn or opened seal",
			"an abnormal region where the package is open",
		],

		"cap open": [
			"a region with an open cap",
			"a region with an unscrewed bottle cap",
			"an abnormal region with a loose cap",
		],

		"broken": [
			"a broken defect region",
			"a region with breakage",
			"an abnormal broken region",
		],

		"deformed": [
			"a deformed defect region",
			"a dented or crushed region",
			"an abnormal region with deformation",
		],

		"missing": [
			"a region with a missing component",
			"a region where a part is absent",
			"an abnormal region with something missing",
		],

		"surface damage": [
			"a surface damage region",
			"a scratched or torn surface region",
			"an abnormal damaged surface region",
		],

		"surface anomaly": [
			"an anomalous surface region",
			"a region with an irregular surface",
			"an abnormal region on the product surface",
		],
	}


def encode_text_with_prompt_ensemble(model, objs, tokenizer, device):
	"""
	Return unified global anchor bank for every object.

	Output:
		text_prompts[obj] = Tensor[D, C]
	where C == len(GLOBAL_DEFECT_CLASSES) for all objects.
	"""

	anchor_prompt_bank = _get_anchor_prompt_bank()

	prompt_templates = [
		"a photo of {}.",
		"a close-up photo of {}.",
		"a clear photo of {}.",
		"a cropped photo of {}.",
		"a detailed photo of {}.",
		"this is {}.",
		"there is {}.",
	]

	global_text_features = []

	for class_name in GLOBAL_DEFECT_CLASSES:
		if class_name not in anchor_prompt_bank:
			raise KeyError(
				f"Class '{class_name}' not found in anchor prompt bank."
			)

		base_prompts = anchor_prompt_bank[class_name]

		prompted_sentences = []
		for base_prompt in base_prompts:
			for template in prompt_templates:
				prompted_sentences.append(template.format(base_prompt))

		tokens = tokenizer(prompted_sentences).to(device)
		class_embeddings = model.encode_text(tokens)
		class_embeddings = class_embeddings / class_embeddings.norm(dim=-1, keepdim=True)

		class_embedding = class_embeddings.mean(dim=0)
		class_embedding = class_embedding / class_embedding.norm()

		global_text_features.append(class_embedding)

	global_text_features = torch.stack(global_text_features, dim=1).to(device)  # [D, C]

	text_prompts = {}
	for obj in objs:
		text_prompts[obj] = global_text_features

	return text_prompts
