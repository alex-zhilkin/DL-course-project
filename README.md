# DL Course Project

Install the dependencies first:

```bash
pip install -e .
```

Download [the data](https://drive.google.com/file/d/1KTWN1Rp-vs5eKmip4tH5tS77y5HUX_CH/view?usp=sharing) and put it in `data/`.

The primary latent-space workflow is now:

- `notebooks/latent_space/04_latent_space_simulator.ipynb` - main mixed/single rollout comparison against the GNN baseline
- `notebooks/latent_space/04a_latent_space_analysis.ipynb` - 2D latent-space interpretation and p-ratio readouts

The older standalone latent simulator notebook is archived at
`notebooks/archive/legacy/04_latent_space_simulator_legacy.ipynb`.

The simple non-neural Reid baseline is:

- `notebooks/08_reid_stiffness_pratio_affine_rollout.ipynb`

Older CV/transformer comparisons are under `notebooks/baselines/`.
Unrelated or superseded experiments are under `notebooks/archive/`.

The repo already includes the model artifacts needed to load the existing results.
So by default the notebooks can reuse those instead of retraining.

If you want to retrain a notebook from scratch, set the flag at the top of that notebook:

- `force_train = True`

Then rerun the notebook.
