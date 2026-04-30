"""ML: Non-Linear Regression - Exponential.

Years held vs property value: a starter home appreciating at ~6 %/yr compounds
exponentially.

    value = a * exp(r * years)

Trick: take ln(y) of the target and fit a regular linear regression on (x, ln(y)).
Recover a = exp(intercept) and r = slope.

Run:
    python 02d_exponential.py
"""

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression

PLOT_PATH = "02d_exponential_plot.png"


def main() -> None:
    data = {
        "years":  [0,   2,   5,   8,   12,  15,  18,  22,  25,  30],
        "value":  [400, 450, 535, 640, 815, 970, 1160, 1465, 1720, 2300],
    }
    test = {"years": [3, 10, 20, 28]}

    df = pd.DataFrame(data)
    print("training data (value in $k):")
    print(df.to_string(index=False))

    ln_value = np.log(df["value"])
    reg = LinearRegression().fit(df[["years"]].values, ln_value)

    r = reg.coef_[0]
    a = float(np.exp(reg.intercept_))
    print(f"\nfit: value = {a:.4f} * exp({r:.6f} * years)")
    print(f"     implied annual appreciation: {(np.exp(r) - 1) * 100:.2f}%")

    tst = pd.DataFrame(test)
    tst["value"] = a * np.exp(r * tst["years"])
    print("\npredictions:")
    print(tst.to_string(index=False))

    year_grid = np.linspace(df.years.min(), df.years.max(), 200)
    curve = a * np.exp(r * year_grid)

    plt.figure()
    plt.xlabel("years held")
    plt.ylabel("property value ($k)")
    plt.scatter(df.years, df.value, color="magenta", marker="*", s=80, label="training")
    plt.plot(year_grid, curve, color="blue", label="exponential fit")
    plt.scatter(tst.years, tst.value, color="red", label="predicted")
    plt.legend()
    plt.savefig(PLOT_PATH, bbox_inches="tight")
    print(f"\nWrote {PLOT_PATH}")


if __name__ == "__main__":
    main()
