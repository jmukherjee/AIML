# Non-Linear Regression - Python (CLI)

Plain-Python CLI ports of the four non-linear regression notebooks in `../jupyter/`.

See [`../README.md`](../README.md) for the story-by-story breakdown of each fit.

## Pre-requisites
- Python 3.9+

## Setup

```sh
cd 01_ML/01_supervised/01_regression/02_nonlinear_reg/python
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

## Run

```sh
python 02a_quadratic.py     # house age   -> price (∪-parabola)
python 02b_cubic.py         # NQI score   -> price (S-curve)
python 02c_logarithmic.py   # lot size    -> price (diminishing returns)
python 02d_exponential.py   # years held  -> value (compound appreciation)
```

Each script prints the training data, the fitted equation, and predictions for a held-out test set, then saves a PNG (`02*_plot.png`) of the data + curve via the headless `Agg` backend.

## How each algorithm uses `sklearn.linear_model.LinearRegression`

- **Quadratic / Cubic** — `PolynomialFeatures(degree=N)` augments the input matrix with `x²` (and `x³`), so the same `LinearRegression` solver returns polynomial coefficients.
- **Logarithmic** — fit on `(ln(x), y)`. The fitted slope and intercept *are* the `a` and `b` of `y = a·ln(x) + b`.
- **Exponential** — fit on `(x, ln(y))`. Recover `a = exp(intercept)`, `r = slope`, then `y = a·exp(r·x)`.
