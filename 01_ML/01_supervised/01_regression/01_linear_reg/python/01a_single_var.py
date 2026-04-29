"""ML: Linear Regression - Single Variate (CLI port of ../jupyter/01a_single_var.ipynb).

Predict house price from house area (sq.ft.).

    y = m*x + b

Run:
    python 01a_single_var.py
"""

import matplotlib

matplotlib.use("Agg")  # non-interactive: write PNG instead of opening a window

import matplotlib.pyplot as plt
import pandas as pd
from sklearn import linear_model

PLOT_PATH = "01a_single_var_plot.png"


def main() -> None:
    data = {
        "area":  [2600, 3000, 3200, 3600, 4000],
        "price": [550000, 565000, 610000, 680000, 725000],
    }
    test = {"area": [2000, 3300, 4100]}

    df = pd.DataFrame(data)
    print("training data:")
    print(df.to_string(index=False))

    reg = linear_model.LinearRegression()
    reg.fit(df[["area"]], df["price"])

    print(f"\ncoef (m):      {reg.coef_[0]}")
    print(f"intercept (b): {reg.intercept_}")

    tst = pd.DataFrame(test)
    tst["price"] = reg.predict(tst[["area"]])
    print("\npredictions:")
    print(tst.to_string(index=False))

    plt.figure()
    plt.xlabel("area")
    plt.ylabel("price")
    plt.scatter(df.area, df.price, color="magenta", marker="*", label="training")
    plt.plot(tst.area, tst.price, color="blue", marker="o", label="predicted")
    plt.legend()
    plt.savefig(PLOT_PATH, bbox_inches="tight")
    print(f"\nWrote {PLOT_PATH}")


if __name__ == "__main__":
    main()
