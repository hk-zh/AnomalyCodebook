def count_parameters(module):
	total = sum(p.numel() for p in module.parameters())
	trainable = sum(p.numel() for p in module.parameters() if p.requires_grad)
	return total, trainable


def print_model_stats(name, module):
	total, trainable = count_parameters(module)
	print(f"{name}:")
	print(f"  total params     = {total:,}")
	print(f"  trainable params = {trainable:,}")