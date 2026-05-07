#!/usr/bin/env bash

# Set this later.
EPSILON_VALUE="0.02"

python guided_sample_diffusion_epsilon_constraint.py \
  --ddpm_ckpt ../../checkpoints/ddpm_epoch_2950.pt \
  --cnn_ckpt ../../checkpoints/cnn_surrogate_best.pt \
  --cnn_cfg config.json \
  --target_json y_target.json \
  --out_dir guided_eps_${EPSILON_VALUE} \
  --device cuda \
  --seed 0 \
  --num 320 \
  --num_batch 32 \
  --pool_mult 3 \
  --max_draws 50000 \
  --volfrac_min 0.30 \
  --binarize_threshold 0.0 \
  --target_mode physical \
  --guidance_scale 2.0 \
  --guidance_start_frac 0.30 \
  --guidance_power 2.0 \
  --guidance_every 1 \
  --grad_clip 1.0 \
  --primary_objective stress \
  --epsilon ${EPSILON_VALUE} \
  --constraint_penalty 10.0 \
  --volfrac_lambda 0.0 \
  --save_every 16 \
  --save_nrow 4
