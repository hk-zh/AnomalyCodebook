# The paper's recipe (seed 42). test.py defaults are the adopted inference
# setting: entropy-adaptive temperature, hard quantization, sigma=4 smoothing,
# multi-scale top-k map score fused with 0.75 x the frozen CLIP global score
# (whole image and its most anomalous 3x3 tile, T=0.1). --global_fusion_weight 0
# gives the map-only image score.
python train.py --dataset mvtec --train_data_path ./data/mvtec --save_path ./exps/stab42/ \
	--epoch 1 --codebook_num_learnable 150 --codebook_init warmup_kmeans \
	--codebook_warmup_steps 100 --revive_every 100 --legacy_revive_at_init --seed 42
python test.py --dataset visa --data_path ./data/visa --checkpoint_path ./exps/stab42/epoch_1.pth \
	--save_path ./results/visa
