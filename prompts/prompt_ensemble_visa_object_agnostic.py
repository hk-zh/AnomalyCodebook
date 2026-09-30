# Copyright (c) 2025 Robert Bosch GmbH
# SPDX-License-Identifier: AGPL-3.0

import torch


def encode_text_with_prompt_ensemble(model, objs, tokenizer, device):
	"""
	Object-agnostic text anchors with a unified global class space.

	Returns:
		text_prompts[obj] = Tensor[D, C]
	where C is fixed for all objects.
	"""

	normal = [
		"normal surface region",
		"defect-free region",
		"intact surface region",
		"flawless region",
		"region without defect",
		"region without damage",
		"clean surface region",
		"regular appearance region",
	]

	damage = [
		"damaged defect region",
		"region with damage",
		"damaged surface region",
		"region showing damage",
		"abnormal damaged region",
		"region with visible wear and tear",
	]

	scratch = [
		"scratch defect region",
		"scratched region",
		"region with scratch",
		"surface region with scratches",
		"abnormal region with scratch marks",
		"region showing scratch defects",
	]

	breakage = [
		"breakage defect region",
		"broken region",
		"region with breakage",
		"surface region with broken parts",
		"abnormal region with breakage",
		"region showing structural breakage",
	]

	burnt = [
		"burnt defect region",
		"burnt region",
		"region with burn marks",
		"scorched surface region",
		"abnormal burnt region",
		"region showing burning defects",
	]

	weird_wick = [
		"weird wick defect region",
		"region with abnormal wick",
		"region with unusual wick shape",
		"abnormal wick region",
		"surface region with wick defect",
		"region showing wick irregularity",
	]

	stuck = [
		"stuck defect region",
		"region with stuck parts",
		"region with adhesion defect",
		"abnormal stuck region",
		"surface region with sticking defect",
		"region showing stuck components",
	]

	crack = [
		"crack defect region",
		"cracked region",
		"region with crack",
		"surface region with cracks",
		"abnormal cracked region",
		"region showing crack lines",
	]

	wrong_place = [
		"misplaced defect region",
		"region with wrong placement",
		"region with misplaced part",
		"abnormal misplaced region",
		"surface region with positioning defect",
		"region showing misalignment",
	]

	partical = [
		"particle defect region",
		"region with particles",
		"region with foreign particles",
		"contaminated particle region",
		"abnormal region with visible particles",
		"surface region with particle contamination",
	]

	bubble = [
		"bubble defect region",
		"region with bubbles",
		"surface region with air bubbles",
		"abnormal bubbly region",
		"region showing bubble defects",
		"region with bubble marks",
	]

	melded = [
		"melded defect region",
		"region with fused material",
		"surface region with melded parts",
		"abnormal melded region",
		"region showing fused areas",
		"region with material fusion defect",
	]

	hole = [
		"hole defect region",
		"region with hole",
		"surface region with puncture",
		"abnormal hole region",
		"region showing perforation",
		"region with visible hole defect",
	]

	melt = [
		"melt defect region",
		"melted region",
		"region with melting defect",
		"abnormal melted region",
		"surface region with melt marks",
		"region showing signs of melting",
	]

	bent = [
		"bent defect region",
		"bent region",
		"region with bending defect",
		"abnormal bent region",
		"surface region with bending",
		"region showing curvature defect",
	]

	spot = [
		"spot defect region",
		"spotted region",
		"region with spots",
		"surface region with visible spots",
		"abnormal spotted region",
		"region showing spotting defects",
	]

	extra = [
		"extra material defect region",
		"region with extra material",
		"region with unwanted addition",
		"abnormal extra-material region",
		"surface region with extra component",
		"region showing additional unwanted pieces",
	]

	chip = [
		"chip defect region",
		"chipped region",
		"region with chip defect",
		"surface region with chipped parts",
		"abnormal chipped region",
		"region showing broken fragments",
	]

	missing = [
		"missing defect region",
		"region with missing part",
		"region with absent component",
		"incomplete region",
		"abnormal missing-part region",
		"surface region with missing pieces",
	]

	prompt_state = [
		normal,
		damage,
		scratch,
		breakage,
		burnt,
		weird_wick,
		stuck,
		crack,
		wrong_place,
		partical,
		bubble,
		melded,
		hole,
		melt,
		bent,
		spot,
		extra,
		chip,
		missing,
	]

	prompt_templates = [
		"a photo of a {}.",
		"a close-up photo of a {}.",
		"a clear photo of a {}.",
		"a cropped photo of a {}.",
		"a detailed photo of a {}.",
		"a good photo of a {}.",
		"a blurry photo of a {}.",
		"a low resolution photo of a {}.",
		"there is a {} in the scene.",
		"this is a {} in the scene.",
	]

	# Build one shared global anchor bank first
	global_text_features = []
	for class_prompts in prompt_state:
		prompted_sentences = []
		for s in class_prompts:
			for template in prompt_templates:
				prompted_sentences.append(template.format(s))

		tokens = tokenizer(prompted_sentences).to(device)
		class_embeddings = model.encode_text(tokens)
		class_embeddings = class_embeddings / class_embeddings.norm(dim=-1, keepdim=True)

		class_embedding = class_embeddings.mean(dim=0)
		class_embedding = class_embedding / class_embedding.norm()
		global_text_features.append(class_embedding)

	global_text_features = torch.stack(global_text_features, dim=1).to(device)  # [D, C]

	# Every object shares the same object-agnostic anchor bank
	text_prompts = {}
	for obj in objs:
		text_prompts[obj] = global_text_features

	return text_prompts