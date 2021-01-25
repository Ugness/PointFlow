#! /bin/bash

input_dim=2
dims="512-512-512"
latent_dims="256-256"
num_blocks=1
latent_num_blocks=1
zdim=128
batch_size=16
lr=2e-3
epochs=4000
ds=multimnist
multimnist_sample_size=500
log_name="gen/multimnist"
resume_checkpoint="checkpoints/gen/multimnist/checkpoint-latest.pt"

python extract_sample_ref.py \
    --resume_checkpoint ${resume_checkpoint} \
    --log_name ${log_name} \
    --lr ${lr} \
    --dataset_type ${ds} \
    --multimnist_sample_size ${multimnist_sample_size} \
    --input_dim ${input_dim} \
    --dims ${dims} \
    --latent_dims ${latent_dims} \
    --num_blocks ${num_blocks} \
    --latent_num_blocks ${latent_num_blocks} \
    --batch_size ${batch_size} \
    --zdim ${zdim} \
    --epochs ${epochs} \
    --save_freq 10 \
    --viz_freq 999999 \
    --log_freq 10 \
    --no_validation \
    --no_writer \
    --val_freq 10 \
    --use_latent_flow \
    --num_workers 0


echo "Done"
exit 0
