# Non-Linear Regression — Jupyter

Four notebooks covering polynomial (quadratic & cubic), logarithmic, and exponential regression — all sharing the same house-price / neighborhood-analytics world used in `../../01_linear_reg/`.

See [`../README.md`](../README.md) for the story-by-story breakdown of each fit.

## Pre-requisites
- Anaconda/Miniconda or Python 3.9-3.12 (the pinned `requirements.txt` versions don't yet support Python 3.13+; if you're on a newer Python use `../python/` instead — same fits, no Jupyter needed)
- Jupyter

## Setup

Same recipe as `../../01_linear_reg/jupyter/README.md`:

```sh
cd 01_ML/01_supervised/01_regression/02_nonlinear_reg/jupyter
virtualenv venv --system-site-packages
source venv/bin/activate
pip install -r requirements.txt
pip install jupyter
python -m ipykernel install --user --name=venv
jupyter notebook
```

Open any of the four notebooks and select the `venv` kernel.

## Notebooks

- `02a_quadratic.ipynb`     — house age vs price (∪-parabola)
- `02b_cubic.ipynb`         — Neighborhood Quality Index vs price (S-curve)
- `02c_logarithmic.ipynb`   — lot size vs price (diminishing returns)
- `02d_exponential.ipynb`   — years held vs property value (compound appreciation)
