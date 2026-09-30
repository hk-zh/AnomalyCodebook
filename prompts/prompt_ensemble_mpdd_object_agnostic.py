# Copyright (c) 2025 Robert Bosch GmbH
# SPDX-License-Identifier: AGPL-3.0

import torch

# Unified global class space for MPDD
GLOBAL_DEFECT_CLASSES = [
	"good",
	"hole",
	"scratch",
	"bent",
	"mismatch",
	"defective painting",
	"rust",
	"flattening",
]


def _get_anchor_prompt_bank():
	"""
	Object-agnostic class prompts.
	All MPDD objects share the same global class space.
	"""

	return {
		"good": [
			"a normal surface region",
			"a defect-free region",
			"an intact surface region",
			"a flawless region",
			"a region without defect",
			"a region in pristine condition",
		],

		"hole": [
			"a hole defect region",
			"a region with a hole",
			"an abnormal region with hole defect",
			"a perforated defect region",
			"a punctured surface region",
			"a region with visible hole",
		],

		"scratch": [
			"a scratch defect region",
			"a scratched surface region",
			"an abnormal region with scratches",
			"a region with scratch marks",
			"a surface region with linear scratches",
			"a damaged region with scratch defect",
		],

		"bent": [
			"a bent defect region",
			"a region with bending defect",
			"an abnormal bent region",
			"a deformed region caused by bending",
			"a warped surface region",
			"a region bent out of shape",
		],

		"mismatch": [
			"a parts mismatch defect region",
			"a region with mismatched parts",
			"an abnormal region with component mismatch",
			"a region with incorrect assembly",
			"a region with misaligned components",
			"a region with part misplacement",
		],

		"defective painting": [
			"a defective painting region",
			"a region with painting defect",
			"an abnormal region with poor paint quality",
			"a surface region with uneven painting",
			"a region with paint imperfection",
			"a region with defective coating",
		],

		"rust": [
			"a rust defect region",
			"a rusty surface region",
			"an abnormal region with rust",
			"a region affected by corrosion",
			"a region with rust spots",
			"a corroded surface region",
		],

		"flattening": [
			"a flattening defect region",
			"a flattened surface region",
			"an abnormal region with flattening",
			"a compressed defect region",
			"a squashed surface region",
			"a region with flattened deformation",
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

		with torch.no_grad():
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