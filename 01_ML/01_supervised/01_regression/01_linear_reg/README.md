# Linear Regression

Two notebooks, each a small house-price story that fits a straight line (or a hyperplane) to the data. Same dataset, same predictions across all three peer folders.

## Stories

| File | Scenario | Inputs (X) | Output (y) | Shape |
|------|----------|------------|------------|-------|
| `01a_single_var` | Predict list price from house area alone — the textbook one-feature fit. | `area` (sq.ft.) | `price` ($) | line: `y = m·x + b` |
| `01b_multi_var`  | Add house age as a second feature; the model learns that older homes shave off price even when area is held constant. | `area` (sq.ft.), `age` (yr) | `price` ($) | plane: `y = m₁·area + m₂·age + b` |

In `01b`, the age coefficient comes out **negative** (~-327 $/yr) — that's the model expressing "each extra year of age trims a few hundred dollars off." Useful intuition for what regression coefficients mean before moving on to non-linear shapes in `../02_nonlinear_reg/`.

## Layout

Three peer folders, same logic in three flavors:

- [`jupyter/`](./jupyter) — original interactive notebooks; matplotlib plots inline.
- [`python/`](./python) — plain CLI scripts (`python 01a_single_var.py`); matplotlib runs headless and saves PNGs.
- [`nodejs/`](./nodejs) — Node.js port using `ml-regression-multivariate-linear`; plots emit Plotly HTML files.

All three produce identical coefficients and predictions — a useful cross-check when first wiring up an ML library in a new language.
