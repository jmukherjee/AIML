# Regression

Six worked examples that walk from a single-variable straight line all the way to compound exponential growth — all framed in one consistent house-price / neighborhood-analytics world so the *shape* of the relationship is what changes between notebooks, not the domain.

Each example is implemented three ways (Jupyter / plain Python CLI / Node.js) under two topic folders:

- [`01_linear_reg/`](./01_linear_reg) — `y = m·x + b`, plus its multi-variable hyperplane variant.
- [`02_nonlinear_reg/`](./02_nonlinear_reg) — quadratic, cubic, logarithmic, exponential — four shapes that all reduce to a linear-regression solver via one trick or another.

## Linear regression

| File | Scenario | Inputs (X) | Output (y) | Shape |
|------|----------|------------|------------|-------|
| [`01a_single_var`](./01_linear_reg) | Predict list price from house area alone — the textbook one-feature fit. | `area` (sq.ft.) | `price` ($) | line: `y = m·x + b` |
| [`01b_multi_var`](./01_linear_reg)  | Add house age as a second feature; the model learns that older homes shave off price even when area is held constant (negative coefficient on age). | `area` (sq.ft.), `age` (yr) | `price` ($) | plane: `y = m₁·area + m₂·age + b` |

## Non-linear regression

| File | Algorithm | Scenario | Input (X) | Output (y) | Curve shape |
|------|-----------|----------|-----------|------------|-------------|
| [`02a_quadratic`](./02_nonlinear_reg)     | Polynomial / Quadratic Curve | New builds priced high, mid-age (15-30 yr) homes dip, very old (50-100 yr) homes recover as historic / character properties. | `age` (yr) | `price` ($k) | ∪-parabola: `y = a·x² + b·x + c` |
| [`02b_cubic`](./02_nonlinear_reg)         | Polynomial / Cubic Curve     | Neighborhood Quality Index (composite of schools, safety, walkability) drives a classic S-curve — flat under ~30, slow rise 30-60, steep premium 60-85, luxury surge 85-100. | `nqi` (0-100) | `price` ($k) | S-curve: `y = a·x³ + b·x² + c·x + d` |
| [`02c_logarithmic`](./02_nonlinear_reg)   | Logarithmic                  | First half-acre adds a lot, every additional acre adds less — diminishing returns from extra land. | `lot_size` (acres) | `price` ($k) | log saturation: `y = a·ln(x) + b` |
| [`02d_exponential`](./02_nonlinear_reg)   | Exponential                  | Compound appreciation: a starter home held for `t` years grows at ~6 % per year. | `years_held` (yr) | `value` ($k) | exponential: `y = a·e^(r·t)` |

## The "one solver, four shapes" trick

`02_nonlinear_reg` deliberately uses the *same* `LinearRegression` solver as `01_linear_reg`, just with the data reshaped:

| Shape | What you change | Why it works |
|-------|-----------------|--------------|
| Polynomial (quadratic / cubic) | Feed `[x, x²]` or `[x, x², x³]` as multiple input columns | A linear fit on extra polynomial columns *is* a polynomial fit on the original `x` |
| Logarithmic | Feed `ln(x)` as the input | If `y = a·ln(x) + b`, then `y` *is* linear in `ln(x)` |
| Exponential | Feed `x` with target `ln(y)`; recover `a = exp(intercept)`, `r = slope` | If `y = a·e^(r·x)`, then `ln(y) = ln(a) + r·x` is linear in `x` |

Useful when porting between languages — even ML libraries with no built-in non-linear regression usually ship a multivariate linear solver, and these four transforms cover the most common curve shapes.

## Layout convention

Each topic folder has three peer language folders that all fit the same data and produce identical coefficients:

```
01_linear_reg/
  README.md         # story-summary table for linear
  jupyter/          # interactive .ipynb notebooks
  python/           # plain CLI scripts (Agg backend, PNG output)
  nodejs/           # Node.js port (Plotly HTML output)

02_nonlinear_reg/
  README.md         # story-summary table for non-linear
  jupyter/
  python/
  nodejs/
```

Cross-checking output across the three implementations is a quick way to catch off-by-one bugs (or library-version drift) when first wiring up a new language.
