# Copyright (c) 2025 Robert Bosch GmbH
# SPDX-License-Identifier: AGPL-3.0
#
# MVTec-LOCO prompts. The benchmark splits anomalies into *structural* ones
# (scratches, cracks, damage: local appearance faults, like classic MVTec) and
# *logical* ones (a missing item, a duplicate, a swapped or misplaced part:
# violations of a rule about the whole scene). Both share one anchor bank here
# so the anomaly map is the sum over every non-good class.

import torch

# Unified global class space for MVTec-LOCO
GLOBAL_DEFECT_CLASSES = [
	"good",
	"missing",
	"additional",
	"misplaced",
	"swapped",
	"broken",
	"crack",
	"scratch",
	"contamination",
	"deformed",
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
			"a correctly assembled region",
		],

		"missing": [
			"a region with a missing component",
			"a region where a part is absent",
			"an abnormal region with something missing",
		],

		"additional": [
			"a region with an extra component",
			"a region with a duplicated part",
			"an abnormal region with an additional object",
		],

		"misplaced": [
			"a misplaced component region",
			"a region with a part in the wrong position",
			"an abnormal region with wrong arrangement",
		],

		"swapped": [
			"a region with the wrong component",
			"a region with a substituted part",
			"an abnormal region with a mismatched object",
		],

		"broken": [
			"a broken defect region",
			"a region with breakage",
			"an abnormal broken region",
		],

		"crack": [
			"a crack defect region",
			"a cracked region",
			"an abnormal region with crack",
		],

		"scratch": [
			"a scratch defect region",
			"a scratched region",
			"an abnormal region with scratch",
		],

		"contamination": [
			"a contamination defect region",
			"a contaminated region",
			"an abnormal region with contamination",
		],

		"deformed": [
			"a deformed defect region",
			"a bent or deformed region",
			"an abnormal region with deformation",
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
