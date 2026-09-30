# Copyright (c) 2025 Robert Bosch GmbH
# SPDX-License-Identifier: AGPL-3.0

import torch

# Unified global class space for MVTec
GLOBAL_DEFECT_CLASSES = [
	"good",
	"bent",
	"broken",
	"color",
	"combined",
	"contamination",
	"crack",
	"cut",
	"fabric",
	"faulty imprint",
	"glue",
	"hole",
	"missing",
	"poke",
	"rough",
	"scratch",
	"squeeze",
	"thread",
	"liquid",
	"misplaced",
	"damaged",
]


def _get_anchor_prompt_bank():
	"""
	Object-agnostic class prompts.
	All objects share the same global class space.
	"""

	return {
		"good": [
			"a normal surface region",
			"a defect-free region",
			"an intact surface region",
			"a flawless region",
		],

		"bent": [
			"a bent defect region",
			"a region with bending defect",
			"an abnormal bent region",
		],

		"broken": [
			"a broken defect region",
			"a region with breakage",
			"an abnormal broken region",
		],

		"color": [
			"a color defect region",
			"a region with color defect",
			"an abnormal discolored region",
		],

		"combined": [
			"a combined defect region",
			"a region with multiple defects",
			"an abnormal region with combined defects",
		],

		"contamination": [
			"a contamination defect region",
			"a contaminated region",
			"an abnormal region with contamination",
		],

		"crack": [
			"a crack defect region",
			"a cracked region",
			"an abnormal region with crack",
		],

		"cut": [
			"a cut defect region",
			"a region with cut defect",
			"an abnormal cut region",
		],

		"fabric": [
			"a fabric defect region",
			"a region with fabric defect",
			"an abnormal fabric region",
		],

		"faulty imprint": [
			"a faulty imprint defect region",
			"a region with print defect",
			"an abnormal region with faulty imprint",
		],

		"glue": [
			"a glue defect region",
			"a region with glue defect",
			"an abnormal region with glue",
		],

		"hole": [
			"a hole defect region",
			"a region with hole defect",
			"an abnormal region with hole",
		],

		"missing": [
			"a missing defect region",
			"a region with missing part",
			"an abnormal region with missing component",
		],

		"poke": [
			"a poke defect region",
			"a region with poke defect",
			"an abnormal poked region",
		],

		"rough": [
			"a rough defect region",
			"a rough surface region",
			"an abnormal rough region",
		],

		"scratch": [
			"a scratch defect region",
			"a scratched region",
			"an abnormal region with scratch",
		],

		"squeeze": [
			"a squeeze defect region",
			"a squeezed region",
			"an abnormal region with squeeze defect",
		],

		"thread": [
			"a thread defect region",
			"a region with thread defect",
			"an abnormal thread region",
		],

		"liquid": [
			"a liquid defect region",
			"a region with liquid contamination",
			"an abnormal liquid-contaminated region",
		],

		"misplaced": [
			"a misplaced defect region",
			"a region with misplacement defect",
			"an abnormal misplaced region",
		],

		"damaged": [
			"a damaged defect region",
			"a damaged region",
			"an abnormal damaged region",
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

	# Build one global embedding bank first
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

	# Every object shares the same global anchor bank
	text_prompts = {}
	for obj in objs:
		text_prompts[obj] = global_text_features

	return text_prompts