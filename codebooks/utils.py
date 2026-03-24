import json
import math
import torch
import torch.nn.functional as F


DEFECT_ALIASES = {
	"broken large": "broken",
	"broken small": "broken",
	"metal contamination": "contamination",
	"cable swap": "combined",
	"cut inner insulation": "cut",
	"cut outer insulation": "cut",
	"missing cable": "missing",
	"missing wire": "missing",
	"poke insulation": "poke",
	"print": "faulty_imprint",
	"fold": "damaged",
	"flip": "damaged",
	"pill type": "combined",
	"manipulated front": "damaged",
	"scratch head": "scratch",
	"scratch neck": "scratch",
	"thread side": "thread",
	"thread top": "thread",
	"oil": "liquid",
	"defective": "damaged",
	"bent lead": "bent",
	"cut lead": "cut",
	"damaged case": "damaged",
	"broken teeth": "broken",
	"fabric border": "fabric",
	"fabric interior": "fabric",
	"split teeth": "broken",
}


def load_prompt_state_dict(json_path):
	"""
	json file should be:
	{
		"good": [...],
		"bent": [...],
		...
	}
	"""
	with open(json_path, "r", encoding="utf-8") as f:
		prompt_state_dict = json.load(f)

	if not isinstance(prompt_state_dict, dict):
		raise ValueError("prompt_state_dict json must be a dict")

	return prompt_state_dict


def normalize_defect_key(defect_name):
	key = defect_name.strip().lower()
	if key in DEFECT_ALIASES:
		return DEFECT_ALIASES[key]
	return key.replace(" ", "_")


def build_scene_templates():
	return [
		'a bad photo of a {}.',
		'a low resolution photo of the {}.',
		'a bad photo of the {}.',
		'a cropped photo of the {}.',
		'a bright photo of a {}.',
		'a dark photo of the {}.',
		'a photo of my {}.',
		'a photo of the cool {}.',
		'a close-up photo of a {}.',
		'a black and white photo of the {}.',
		'a bright photo of the {}.',
		'a cropped photo of a {}.',
		'a jpeg corrupted photo of a {}.',
		'a blurry photo of the {}.',
		'a photo of the {}.',
		'a good photo of the {}.',
		'a photo of one {}.',
		'a close-up photo of the {}.',
		'a photo of a {}.',
		'a low resolution photo of a {}.',
		'a photo of a large {}.',
		'a blurry photo of a {}.',
		'a jpeg corrupted photo of the {}.',
		'a good photo of a {}.',
		'a photo of the small {}.',
		'a photo of the large {}.',
		'a black and white photo of a {}.',
		'a dark photo of a {}.',
		'a photo of a cool {}.',
		'a photo of a small {}.',
		'there is a {} in the scene.',
		'there is the {} in the scene.',
		'this is a {} in the scene.',
		'this is the {} in the scene.',
		'this is one {} in the scene.'
	]


def generate_product_defect_prompts(product_name, defect_name, prompt_state_dict):
	"""
	只根据 prompt_state_dict 生成某个 (product, defect) 的 prompts

	Args:
		product_name: e.g. 'bottle'
		defect_name: e.g. 'broken small'
		prompt_state_dict: loaded from json

	Returns:
		prompts: list[str]
		prompt_key: normalized key used in prompt_state_dict
	"""
	prompt_key = normalize_defect_key(defect_name)

	if prompt_key not in prompt_state_dict:
		return [], prompt_key

	defect_templates = prompt_state_dict[prompt_key]
	scene_templates = build_scene_templates()

	prompts = []
	for defect_template in defect_templates:
		phrase = defect_template.format(product_name)
		for scene_template in scene_templates:
			prompts.append(scene_template.format(phrase))

	return prompts, prompt_key


def build_multi_prototypes(embeddings, num_prototypes=4):
	"""
	Args:
		embeddings: [P, D]
	Returns:
		prototypes: [M, D], M <= num_prototypes
	"""
	if embeddings.ndim != 2:
		raise ValueError(f"embeddings must be 2D, got {embeddings.shape}")

	P, D = embeddings.shape
	if P == 0:
		raise ValueError("empty embeddings")

	embeddings = F.normalize(embeddings, dim=-1)

	if P <= num_prototypes:
		return embeddings

	chunk_size = math.ceil(P / num_prototypes)
	prototypes = []

	for start in range(0, P, chunk_size):
		chunk = embeddings[start:start + chunk_size]
		proto = chunk.mean(dim=0)
		proto = F.normalize(proto, dim=-1)
		prototypes.append(proto)

	return torch.stack(prototypes, dim=0)


def get_semantic_codes_for_pairs(
	model,
	dataset,
	tokenizer,
	device,
	prompt_json_path,
	num_prototypes=4
):
	"""
	只根据 prompt_state_dict + 你给定的 (product, defect) pairs 生成 semantic codes

	Args:
		product_defect_pairs: list of tuples
			e.g.
			[
				("bottle", "broken small"),
				("bottle", "contamination"),
				("cable", "bent")
			]

	Returns:
		semantic_codes: [N, D]
		code_meta: list[dict]
	"""
	prompt_state_dict = load_prompt_state_dict(prompt_json_path)

	semantic_codes = []
	code_meta = []

	model.eval()
	with torch.no_grad():
		for product_name, defect_name in product_defect_pairs:
			prompts, prompt_key = generate_product_defect_prompts(
				product_name=product_name,
				defect_name=defect_name,
				prompt_state_dict=prompt_state_dict
			)

			if len(prompts) == 0:
				print(f"[Warning] no prompts for ({product_name}, {defect_name}) -> key '{prompt_key}', skip.")
				continue

			tokens = tokenizer(prompts).to(device)
			text_features = model.encode_text(tokens)	# [P, D]
			text_features = F.normalize(text_features, dim=-1)

			prototypes = build_multi_prototypes(
				text_features,
				num_prototypes=num_prototypes
			)

			semantic_codes.append(prototypes)

			for proto_idx in range(prototypes.size(0)):
				code_meta.append({
					"product": product_name,
					"defect": defect_name,
					"prompt_key": prompt_key,
					"prototype_idx": proto_idx,
					"num_prototypes_for_pair": prototypes.size(0),
					"num_prompts_for_pair": len(prompts)
				})

	if len(semantic_codes) == 0:
		raise ValueError("No semantic codes generated.")

	semantic_codes = torch.cat(semantic_codes, dim=0)	# [N, D]
	return semantic_codes, code_meta