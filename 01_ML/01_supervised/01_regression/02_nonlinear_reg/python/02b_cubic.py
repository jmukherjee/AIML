"""ML: Non-Linear Regression - Polynomial / Cubic.

Neighborhood Quality Index (0-100) vs price: rough zones are flat-low,
working-class areas rise slowly, middle/upper-middle suburbs climb steeply,
luxury enclaves surge — an S-curve a cubic captures.

    price = a * nqi^3 + b * nqi^2 + c * nqi + d

Run:
    python 02b_cubic.py
"""

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures

PLOT_PATH = "02b_cubic_plot.png"


def main() -> None:
    data = {
        "nqi":   [10,  20,  30,  40,  50,  60,  70,  80,  85,  90,  95,  100],
        "price": [180, 195, 215, 250, 310, 410, 560, 760, 900, 1080, 1320, 1600],
    }
    test = {"nqi": [25, 55, 75, 92]}

    df = pd.DataFrame(data)
    print("training data (price in $k):")
    print(df.to_string(index=False))

    poly = PolynomialFeatures(degree=3, include_bias=False)
    X = poly.fit_transform(df[["nqi"]].values)
    reg = LinearRegression().fit(X, df["price"])

    # PolynomialFeatures returns columns in order [x, x^2, x^3]
    c1, c2, c3 = reg.coef_
    d = reg.intercept_
    print(f"\nfit: price = {c3:.4f} * nqi^3 + {c2:.4f} * nqi^2 + {c1:.4f} * nqi + {d:.4f}")

    tst = pd.DataFrame(test)
    tst["price"] = reg.predict(poly.transform(tst[["nqi"]].values))
    print("\npredictions:")
    print(tst.to_string(index=False))

    nqi_grid = np.linspace(df.nqi.min(), df.nqi.max(), 200)
    curve = reg.predict(poly.transform(nqi_grid.reshape(-1, 1)))

    plt.figure()
    plt.xlabel("Neighborhood Quality Index")
    plt.ylabel("price ($k)")
    plt.scatter(df.nqi, df.price, color="magenta", marker="*", s=80, label="training")
    plt.plot(nqi_grid, curve, color="blue", label="cubic fit")
    plt.scatter(tst.nqi, tst.price, color="red", label="predicted")
    plt.legend()
    plt.savefig(PLOT_PATH, bbox_inches="tight")
    print(f"\nWrote {PLOT_PATH}")


if __name__ == "__main__":
    main()
