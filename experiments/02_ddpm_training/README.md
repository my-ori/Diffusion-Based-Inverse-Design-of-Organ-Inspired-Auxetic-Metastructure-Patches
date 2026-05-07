# Auxetic Unit-Cell Diffusion (DDPM)

This is a small, **unconditional** diffusion model (DDPM) that learns to generate new 50×50 unit-cell images.

It **only uses the images `X`** from your `.npz` dataset (the same dataset used for your CNN).
The stress/Poisson curves are ignored during diffusion training.

## Key requirement: volume fraction > 30%
We satisfy it in two places:

1) **Training data filter**: keep only samples with `solid_fraction >= 0.30`  
2) **Sampling filter**: rejection sampling to keep only generated samples with `solid_fraction >= 0.30`

> solid_fraction = (# pixels with value 1) / (50*50)

---

## Install
You need:
- Python 3.9+
- PyTorch
- NumPy
- Pillow

Example:
```bash
pip install torch numpy pillow
```

## Train
Run from the folder that contains:
- `train_diffusion.py`
- `sample_diffusion.py`
- `data_loader_npz.py`  (your existing loader)
- `diffusion/` folder

Example:
```bash
python train_diffusion.py \
  --npz_path ../../data/auxetic_unitcell_dataset.npz \
  --out_dir runs/ddpm1 \
  --batch_size 64 \
  --epochs 50 \
  --volfrac_min 0.30
```

Outputs:
- `runs/ddpm1/checkpoints/*.pt`
- `runs/ddpm1/samples/epoch_*.png`
- `runs/ddpm1/dataset_info.json`

## Sample / Generate
```bash
python sample_diffusion.py \
  --ckpt runs/ddpm1/checkpoints/ddpm_epoch_049.pt \
  --out_dir runs/ddpm1/gen \
  --num 500 \
  --volfrac_min 0.30
```

Outputs:
- `grid_64.png` (preview)
- `samples_500.pt` (raw in [-1,1])
- `samples_500_bin01.npy` (binarized 0/1)

---

## Tips
- If you get **OOM**: reduce `--batch_size` during training.
- If sampling acceptance is low (too many samples rejected):
  - Increase `--num_batch`
  - Slightly reduce `--volfrac_min` (e.g., 0.28) OR
  - Retrain with `--volfrac_min 0.30` but a bit more capacity (`--base_channels 96`)


## Why previews may look like pure noise
- The original training script saved **binarized** previews; early in training this often looks like random noise.
- This v2 script saves both `*_gray.png` and `*_bin.png` and adds `--t_sampling early`.
- Also note: with ~2323 images and batch_size=64 you only get ~36 steps/epoch; 50 epochs ≈ 1800 steps, often too few.

Recommended run:
```bash
python train_diffusion.py \
  --npz_path ../../data/auxetic_unitcell_dataset.npz \
  --out_dir ddpm1 \
  --batch_size 64 \
  --timesteps 400 --schedule linear \
  --t_sampling early \
  --epochs 500 --sample_every 50 --save_every 50
```

The generated checkpoint files are ignored by Git. To use a trained DDPM checkpoint with guided sampling, copy or rename it locally as:

```text
../../checkpoints/ddpm_epoch_2950.pt
```
