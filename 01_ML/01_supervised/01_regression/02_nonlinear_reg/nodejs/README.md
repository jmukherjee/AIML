# Non-Linear Regression - Node.js

Node.js port of the four notebooks in `../jupyter/`. Same data, same coefficients as `../python/`.

See [`../README.md`](../README.md) for the story-by-story breakdown of each fit.

## Pre-requisites
- Node.js >= 18

## Setup

```sh
cd 01_ML/01_supervised/01_regression/02_nonlinear_reg/nodejs
npm install
```

## Run

```sh
npm run quadratic     # 02a — house age   -> price
npm run cubic         # 02b — NQI score   -> price
npm run logarithmic   # 02c — lot size    -> price
npm run exponential   # 02d — years held  -> value
```

Each script prints the training data, the fitted equation, the test-set predictions, and writes a Plotly HTML file (`02*_plot.html`) — open in a browser.

## How each algorithm uses one linear solver

`ml-regression-multivariate-linear` only knows how to fit a hyperplane. To get four non-linear shapes out of it:

| Algorithm | Feature/target trick |
| --------- | -------------------- |
| Quadratic | Feed `[x, x²]` as columns; the returned weights are `[b, a, c]` for `y = a·x² + b·x + c`. |
| Cubic     | Feed `[x, x², x³]`; weights are `[c, b, a, d]` for `y = a·x³ + b·x² + c·x + d`. |
| Logarithmic | Feed `ln(x)`; weights are exactly `[a, b]` for `y = a·ln(x) + b`. |
| Exponential | Feed `x` with **target** `ln(y)`; recover `a = exp(intercept)`, `r = slope`, then `y = a·exp(r·x)`. |
