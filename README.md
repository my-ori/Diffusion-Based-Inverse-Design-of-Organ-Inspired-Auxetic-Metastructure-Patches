# Diffusion-Based Inverse Design of Organ-Inspired Auxetic Metastructure Patches

The code implements a data-driven inverse design workflow for auxetic patch unit cells:

1. Train a CNN surrogate to predict stress-strain and strain-dependent Poisson's-ratio response curves from 50 x 50 binary unit-cell images.
2. Train an unconditional DDPM geometry generator on feasible unit-cell images.
3. Use CNN-guided DDPM sampling to generate designs matching target mechanical-response curves.
4. Use an epsilon-constraint sweep to construct multi-objective stress/Poisson trade-off candidates.

## Repository Contents
```
data/
  auxetic_unitcell_dataset.npz
experiments/
  01_cnn_surrogate/
    CNN surrogate training code and run command.
  02_ddpm_training/
    DDPM training/sampling code and run command.
  03_guided_testset_case/
    Single-target guided DDPM test case, target curve, generated candidates, and plots.
  04_multi_objective_epsilon_sweep/
    Epsilon-constraint guided DDPM sweep code, target curve, and Pareto plot.
checkpoints/
  README.md
```

See [DATA_DICTIONARY.md](DATA_DICTIONARY.md) for the packaged dataset fields and output artifact descriptions.


## Checkpoints and GitHub Upload

Large PyTorch checkpoint files (`*.pt`) are intentionally excluded from the GitHub upload. 

## Environment

Python 3.9+ is recommended. Install the Python dependencies with:

```bash
python -m pip install -r requirements.txt
```

Main dependencies:

- PyTorch
- NumPy
- pandas
- matplotlib
- Pillow

GPU acceleration is recommended for training and guided sampling. The scripts fall back to CPU when CUDA is unavailable, but full DDPM training/sampling will be slow on CPU.

## Reproducing the Main Experiments

Run commands are preserved next to each experiment. The shortest starting points are:

```bash
cd experiments/01_cnn_surrogate
python train_surrogate.py --npz ../../data/auxetic_unitcell_dataset.npz --out_dir run1 --val_ratio 0.05 --test_ratio 0.05 --holdout_classes double_sin --seed 600 --volfrac_min 0.15 --blocks "2,3,4" --head_hidden_dim 256 --dropout_head 0.1 --epochs 3000 --lr 5e-4
```

```bash
cd experiments/02_ddpm_training
python train_diffusion.py --npz_path ../../data/auxetic_unitcell_dataset.npz --out_dir ddpm1 --seed 60 --volfrac_min 0.15 --timesteps 500 --schedule linear --t_sampling uniform --epochs 3000 --save_every 50 --sample_every 50 --num_sample 36 --sample_nrow 6 --holdout_classes double_sin
```

```bash
cd experiments/03_guided_testset_case
python guided_sample_diffusion.py --ddpm_ckpt ../../checkpoints/ddpm_epoch_2950.pt --cnn_ckpt ../../checkpoints/cnn_surrogate_best.pt --cnn_cfg config.json --target_json y_target.json --out_dir guided_y1 --num 320 --num_batch 32 --pool_mult 3 --volfrac_min 0.10 --guidance_scale 2.0 --guidance_start_frac 0.30 --guidance_power 2.0 --w_stress 1.0 --w_nu 1.0
```

```bash
cd experiments/04_multi_objective_epsilon_sweep
python sweep_epsilon_runs_and_merge.py --sampler_py guided_sample_diffusion_epsilon_constraint.py --ddpm_ckpt ../../checkpoints/ddpm_epoch_2950.pt --cnn_ckpt ../../checkpoints/cnn_surrogate_best.pt --cnn_cfg config.json --target_json y_target.json --out_dir sweep_eps_y_nu --primary_objective stress --epsilons 0.03,0.05,0.0835,0.1392,0.2323,0.3875,0.6463,1.0781,1.7985,3.0 --runs_per_epsilon 10 --base_seed 200 --num 1 --num_batch 128 --pool_mult 1 --max_draws 50000 --volfrac_min 0.10 --guidance_scale 2.0 --guidance_start_frac 0.30 --guidance_power 2.0 --constraint_penalty 10.0
```

## Packaged Results

The repository includes the following reusable outputs:

- Dataset: `data/auxetic_unitcell_dataset.npz`
- Guided test-set candidates: `experiments/03_guided_testset_case/generated_designs/`
- Multi-objective Pareto plot: `experiments/04_multi_objective_epsilon_sweep/sweep_eps_y_nu/merged/pareto_scatter_all100.png`


