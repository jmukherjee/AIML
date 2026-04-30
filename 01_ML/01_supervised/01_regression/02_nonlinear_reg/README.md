# Non-Linear Regression

Four notebooks that pick up where `../01_linear_reg/` leaves off — same house-price / neighborhood-analytics world, but each story is shaped to make a different non-linear curve the *natural* fit. Each algorithm is implemented across all three peer folders (`jupyter/`, `python/`, `nodejs/`).

## Stories

| File | Algorithm | Scenario | Input (X) | Output (y) | Curve shape |
|------|-----------|----------|-----------|------------|-------------|
| `02a_quadratic`     | Polynomial / Quadratic Curve | New builds sell at premium, mid-age (15-30 yr) homes dip as finishes date and no heritage value yet, very old homes (50-100 yr) recover as historic/character properties. | `age` (yr) | `price` ($) | ∪-parabola: `y = a·x² + b·x + c` |
| `02b_cubic`         | Polynomial / Cubic Curve     | Neighborhood Quality Index (composite of schools, safety, walkability) drives a classic S-curve: flat under ~30 (rough areas), slow rise 30-60 (working-class), steep premium 60-85 (middle/upper-middle), luxury surge 85-100. | `nqi` (0-100) | `price` ($k) | S-curve: `y = a·x³ + b·x² + c·x + d` |
| `02c_logarithmic`   | Logarithmic                  | First half-acre adds a lot of value, every additional acre adds less — diminishing returns from extra land. | `lot_size` (acres) | `price` ($k) | log saturation: `y = a·ln(x) + b` |
| `02d_exponential`   | Exponential                  | Compound appreciation: a starter home held for `t` years grows at ~6 % per year. | `years_held` (yr) | `value` ($k) | exponential: `y = a·e^(r·t)` |

## How the fits actually work

Two of these fit through a linear-regression library by using a feature/target transform — useful trick to know:

- **Polynomial (`02a`, `02b`)** — augment the input matrix with `x²`, `x³`, … and let linear regression solve for the polynomial coefficients. Python uses `sklearn.preprocessing.PolynomialFeatures`; Node.js builds the same matrix by hand and feeds `ml-regression-multivariate-linear`.
- **Logarithmic (`02c`)** — fit a straight line on `(ln(x), y)`. The slope and intercept are the `a` and `b` of `y = a·ln(x) + b`.
- **Exponential (`02d`)** — fit a straight line on `(x, ln(y))`. Take `a = exp(intercept)` and `r = slope` to recover `y = a·e^(r·x)`.

The same trick lets you reuse one linear solver to fit four non-linear shapes — handy when porting between languages with different ML library coverage.

## Layout

- [`jupyter/`](./jupyter) — interactive notebooks with matplotlib plots inline.
- [`python/`](./python) — plain CLI scripts; matplotlib runs headless and saves PNGs.
- [`nodejs/`](./nodejs) — Node.js port; plots emit Plotly HTML files.

All three folders fit identical coefficients on identical synthetic data, so you can cross-check across languages while learning the algorithms.
