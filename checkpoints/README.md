# Local Checkpoints

Large trained PyTorch checkpoint files are intentionally not included in the GitHub upload.

If you want to rerun guided sampling without retraining, place local checkpoint files here, for example:

```text
checkpoints/
  cnn_surrogate_best.pt
  ddpm_epoch_2950.pt
```

You can also generate these files by rerunning:

- `experiments/01_cnn_surrogate/train_surrogate.py`
- `experiments/02_ddpm_training/train_diffusion.py`

The repository `.gitignore` excludes `*.pt` files so these local model weights are not committed accidentally.

