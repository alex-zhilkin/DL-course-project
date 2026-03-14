# DL Course Project

Install the dependencies first:

```bash
pip install -e .
```

Download [the data](https://drive.google.com/file/d/1KTWN1Rp-vs5eKmip4tH5tS77y5HUX_CH/view?usp=sharing) and put it in `data/`.

Then run the notebooks in `notebooks/`.
The main ones are:

- `notebooks/01_train_cv_transformer.ipynb`
- `notebooks/02_train_chignolin_cv_transformer.ipynb`
- `notebooks/03_training_hybrid.ipynb`

The repo already includes the model artifacts needed to load the existing results.
So by default the notebooks can reuse those instead of retraining.

If you want to retrain a notebook from scratch, set the flag at the top of that notebook:

- `force_train = True`

Then rerun the notebook.
