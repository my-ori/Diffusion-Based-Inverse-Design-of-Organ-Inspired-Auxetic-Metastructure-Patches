# CNN Surrogate Training

This folder contains the CNN surrogate model and training code for predicting stress-strain and Poisson's-ratio curves from binary 50 x 50 unit-cell images.

Run from this folder:

```bash
python train_surrogate.py \
  --npz ../../data/auxetic_unitcell_dataset.npz \
  --out_dir run1 \
  --val_ratio 0.05 \
  --test_ratio 0.05 \
  --holdout_classes double_sin \
  --seed 600 \
  --volfrac_min 0.15 \
  --blocks "2,3,4" \
  --head_hidden_dim 256 \
  --dropout_head 0.1 \
  --epochs 3000 \
  --lr 5e-4
```

The trained checkpoint `run1/best.pt` is ignored by Git. To use it with guided sampling, copy or rename it locally as:

```text
../../checkpoints/cnn_surrogate_best.pt
```

