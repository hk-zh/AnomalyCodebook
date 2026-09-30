import json
import torch
import torch.nn.functional as F


def load_prompt_config(json_path: str, dataset_name: str):
	with open(json_path, "r", encoding="utf-8") as f:
		cfg = json.load(f)

	prompt_templates = cfg["prompt_template"]
	defect_prompts = cfg[dataset_name]["defect_prompts"]
	product2defects = cfg[dataset_name]["product2defects"]

	return prompt_templates, defect_prompts, product2defects


def generate_all_prompt_strings(json_path: str, dataset_name: str, objs):
	prompt_templates, defect_prompts, product2defects = load_prompt_config(
		json_path=json_path,
		dataset_name=dataset_name
	)

	all_prompts = []

	for product_name in objs:
		if product_name not in product2defects:
			print(f"[Warning] product '{product_name}' not found, skip.")
			continue

		for defect_name in product2defects[product_name]:
			if defect_name not in defect_prompts:
				print(f"[Warning] defect '{defect_name}' not found, skip.")
				continue

			for defect_prompt in defect_prompts[defect_name]:
				phrase = defect_prompt.format(product_name)
				for scene_template in prompt_templates:
					all_prompts.append(scene_template.format(phrase))

	return all_prompts


def get_all_text_prompt_embeddings(
	model,
	objs,
	tokenizer,
	device,
	json_path: str = "./codebooks/prompts.json",
	dataset_name: str = "mvtec_ad",
	normalize: bool = True,
	text_batch_size: int = 32
):
	"""
	Return:
		text_embeddings: [N, D]
	"""
	all_prompts = generate_all_prompt_strings(
		json_path=json_path,
		dataset_name=dataset_name,
		objs=objs
	)

	if len(all_prompts) == 0:
		raise ValueError("No prompts generated.")

	model.eval()
	all_embeddings = []

	with torch.no_grad():
		for start in range(0, len(all_prompts), text_batch_size):
			end = start + text_batch_size
			prompt_batch = all_prompts[start:end]

			tokens = tokenizer(prompt_batch).to(device)
			embeddings = model.encode_text(tokens)   # [B, D]

			if normalize:
				embeddings = F.normalize(embeddings, dim=-1)

			all_embeddings.append(embeddings)

	text_embeddings = torch.cat(all_embeddings, dim=0)  # [N, D]
	return text_embeddings


def get_averaged_text_prompt_embeddings(
	model,
	objs,
	tokenizer,
	device,
	json_path: str = "./codebooks/prompts.json",
	dataset_name: str = "mvtec_ad",
	normalize: bool = True,
	text_batch_size: int = 32,
	return_meta: bool = False,
):
	"""
	Build one averaged text embedding for each (product, defect) pair.

	Args:
		model: CLIP model
		objs: list of product names
		tokenizer: CLIP tokenizer
		device: cuda or cpu
		json_path: prompt config json path
		dataset_name: e.g. "mvtec_ad" or "visa"
		normalize: whether to normalize prompt embeddings and final averaged embedding
		text_batch_size: batch size for text encoder
		return_meta: whether to also return metadata for each embedding

	Returns:
		text_embeddings: Tensor [N, D]
			N = total number of (product, defect) pairs
		meta: optional list of dicts
	"""
	prompt_templates, defect_prompts, product2defects = load_prompt_config(
		json_path=json_path,
		dataset_name=dataset_name
	)

	if len(objs) == 0:
		raise ValueError("objs is empty.")

	model.eval()
	all_pair_embeddings = []
	meta = []

	with torch.no_grad():
		for product_name in objs:
			if product_name not in product2defects:
				print(f"[Warning] product '{product_name}' not found, skip.")
				continue

			for defect_name in product2defects[product_name]:
				if defect_name not in defect_prompts:
					print(f"[Warning] defect '{defect_name}' not found, skip.")
					continue

				prompts = []
				for defect_prompt in defect_prompts[defect_name]:
					phrase = defect_prompt.format(product_name)
					for scene_template in prompt_templates:
						prompts.append(scene_template.format(phrase))

				if len(prompts) == 0:
					print(f"[Warning] no prompts generated for ({product_name}, {defect_name}), skip.")
					continue

				prompt_embeddings = []
				for start in range(0, len(prompts), text_batch_size):
					end = start + text_batch_size
					prompt_batch = prompts[start:end]

					tokens = tokenizer(prompt_batch).to(device)
					embeddings = model.encode_text(tokens)  # [B, D]

					if normalize:
						embeddings = F.normalize(embeddings, dim=-1)

					prompt_embeddings.append(embeddings)

				prompt_embeddings = torch.cat(prompt_embeddings, dim=0)  # [P, D]

				# Average over all prompts for this (product, defect)
				pair_embedding = prompt_embeddings.mean(dim=0)  # [D]

				if normalize:
					pair_embedding = F.normalize(pair_embedding, dim=-1)

				all_pair_embeddings.append(pair_embedding)

				if return_meta:
					meta.append({
						"product": product_name,
						"defect": defect_name,
						"num_prompts": len(prompts),
					})

	if len(all_pair_embeddings) == 0:
		raise ValueError("No averaged text embeddings generated.")

	text_embeddings = torch.stack(all_pair_embeddings, dim=0)  # [N, D]

	if return_meta:
		return text_embeddings, meta

	return text_embeddings
