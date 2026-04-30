"""ML: Non-Linear Regression - Logarithmic.

Lot size (acres) vs price: the first half-acre adds a lot, every additional
acre adds less — a classic log-saturation curve fit by transforming x.

    price = a * ln(lot_size) + b

Trick: take ln(x) of the input and fit a regular linear regression on (ln(x), y).
The slope and intercept are exactly the a and b above.

Run:
    python 02c_logarithmic.py
"""

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression

PLOT_PATH = "02c_logarithmic_plot.png"


def main() -> None:
    data = {
        "lot_size": [0.10, 0.20, 0.35, 0.50, 0.75, 1.00, 1.50, 2.00, 3.00, 5.00],
        "price":    [620,  690,  760,  800,  850,  890,  940,  975,  1020, 1080],
    }
    test = {"lot_size": [0.25, 0.80, 2.50, 4.00]}

    df = pd.DataFrame(data)
    print("training data (price in $k):")
    print(df.to_string(index=False))

    ln_lot = np.log(df[["lot_size"]].values)
    reg = LinearRegression().fit(ln_lot, df["price"])

    a = reg.coef_[0]
    b = reg.intercept_
    print(f"\nfit: price = {a:.4f} * ln(lot_size) + {b:.4f}")

    tst = pd.DataFrame(test)
    tst["price"] = reg.predict(np.log(tst[["lot_size"]].values))
    print("\npredictions:")
    print(tst.to_string(index=False))

    lot_grid = np.linspace(df.lot_size.min(), df.lot_size.max(), 200)
    curve = a * np.log(lot_grid) + b

    plt.figure()
    plt.xlabel("lot size (acres)")
    plt.ylabel("price ($k)")
    plt.scatter(df.lot_size, df.price, color="magenta", marker="*", s=80, label="training")
    plt.plot(lot_grid, curve, color="blue", label="log fit")
    plt.scatter(tst.lot_size, tst.price, color="red", label="predicted")
    plt.legend()
    plt.savefig(PLOT_PATH, bbox_inches="tight")
    print(f"\nWrote {PLOT_PATH}")


if __name__ == "__main__":
    main()
