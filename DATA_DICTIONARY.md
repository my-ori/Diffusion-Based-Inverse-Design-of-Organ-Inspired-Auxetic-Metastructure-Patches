# Data Dictionary

## Dataset Files

The dataset is stored once at:

- `data/auxetic_unitcell_dataset.npz`

Each `.npz` file contains:

| Field | Shape | Type | Description |
| --- | ---: | --- | --- |
| `X` | `(3128, 50, 50)` | `uint8` | Binary unit-cell geometry images. Solid pixels are encoded as 1 and void pixels as 0. |
| `y_stress` | `(3128, 30)` | `float32` | Stress-strain response sampled at 30 strain levels. |
| `y_nu` | `(3128, 30)` | `float32` | Strain-dependent Poisson's-ratio response sampled at 30 strain levels. |
| `class_ids` | `(3128,)` | `int64` | Integer class index for each unit-cell geometry. |
| `class_names` | `(9,)` | `object` | Class-name lookup table for `class_ids`. |
| `sample_keys` | `(3128,)` | `object` | Human-readable sample identifiers derived from geometry class and parameters. |
| `target_strains` | `(30,)` | `float32` | Strain sample locations for `y_stress` and `y_nu`. |

The packaged class names are:

```text
Lozenge
Oval
Peanut
Reactangular
anti_chiral_iso
double_sin
lozenge_chiral
sinusoidal
tetra_chiral
```

The CNN training command uses `volfrac_min=0.15`, which retained 2968 samples in the original training run.

## Model and Result Artifacts

| Path | Description |
| --- | --- |
| `experiments/03_guided_testset_case/y_target.json` | Target response curve for the single guided DDPM test case. |
| `experiments/03_guided_testset_case/generated_designs/` | Guided DDPM output images and metadata retained for publication. |
| `experiments/04_multi_objective_epsilon_sweep/y_target.json` | Target response curve for the multi-objective guided DDPM case. |
| `experiments/04_multi_objective_epsilon_sweep/sweep_eps_y_nu/merged/pareto_scatter_all100.png` | Retained Pareto scatter plot from the epsilon sweep. |

## Notes for Reuse

- `*.pt` files are intentionally ignored and not included in the GitHub upload.
- `target_json` files accept physical response values by default in the guided sampling scripts.
- The sampling scripts use the normalization statistics stored in the CNN `config.json`.
- GPU execution is recommended for guided DDPM sampling.
