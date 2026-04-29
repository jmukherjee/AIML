# Linear Regression - Python (CLI)

Plain-Python CLI port of the notebooks in `../jupyter/`. Same data, same predictions, no Jupyter.

## Pre-requisites
- Python 3.9+

## Setup

```sh
cd 01_ML/01_supervised/01_regression/01_linear_reg/python
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

## Run

```sh
python 01a_single_var.py   # one feature  (area)
python 01b_multi_var.py    # two features (area, age)
```

Each script prints the input table, the fitted coefficients/intercept, and the predicted prices for the test rows. Each also saves a PNG (`01a_single_var_plot.png` / `01b_multi_var_plot.png`) — matplotlib runs headless via the `Agg` backend, so no display is required.

## Relationship to the Jupyter version

The code mirrors `../jupyter/*.ipynb` cell-for-cell with three differences:

1. `matplotlib.use("Agg")` is set before importing `pyplot` so the script works over SSH / in CI without a display.
2. Plots are written to PNG via `savefig` instead of being shown inline.
3. There is no `pip install -r requirements.txt` magic; install deps once during setup above.
