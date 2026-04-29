# Linear Regression - Node.js

Node.js port of the notebooks in `../jupyter/`. Same data, same predictions. The CLI Python equivalent lives in `../python/`.

## Pre-requisites
- Node.js >= 18

## Setup

```sh
cd 01_ML/01_supervised/01_regression/01_linear_reg/nodejs
npm install
```

## Run

```sh
npm run single   # 01a_single_var.js — one feature (area)
npm run multi    # 01b_multi_var.js  — two features (area, age)
```

Each script prints the input table, the fitted coefficients/intercept, and the predicted prices for the test rows. Each also writes an HTML file (`01a_single_var_plot.html` / `01b_multi_var_plot.html`) — open it in a browser to see the regression line / regression plane rendered with Plotly.

## Mapping to the Jupyter notebooks

| Python (`sklearn`)         | Node.js (`ml-regression-multivariate-linear`) |
| :------------------------- | :-------------------------------------------- |
| `linear_model.LinearRegression()` + `.fit(X, y)` | `new MLR(X, Y)` (constructor fits on the spot) |
| `reg.coef_`                | `reg.weights[0..n-1]` (one row per feature)   |
| `reg.intercept_`           | `reg.weights[n]` (last row)                   |
| `reg.predict(X)`           | `reg.predict(X)`                              |
| `matplotlib` / `mpl_toolkits.mplot3d` | Plotly via static HTML (`plot.js`)   |

`MLR` always works on 2D arrays (one row per sample, one column per feature/output), so even the single-variate case wraps each scalar into a 1-element array.
