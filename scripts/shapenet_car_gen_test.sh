#! /bin/bash

python test.py \
    --cates car \
    --resume_checkpoint pretrained_models/gen/car/checkpoint.pt \
    --dims 512-512-512 \
    --latent_dims 256-256 \
    --use_latent_flow \
	# --standardize_per_shape


