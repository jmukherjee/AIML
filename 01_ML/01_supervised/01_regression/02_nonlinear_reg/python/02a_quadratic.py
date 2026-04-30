"""ML: Non-Linear Regression - Polynomial / Quadratic.

House age vs price: brand-new homes priced high, mid-age homes priced low,
heritage homes priced high again — a U-shape that a parabola fits cleanly.

    price = a * age^2 + b * age + c

Run:
    python 02a_quadratic.py
"""

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures

PLOT_PATH = "02a_quadratic_plot.png"


def main() -> None:
    data = {
        "age":   [1,   5,   10,  15,  20,  25,  30,  40,  55,  70,  85,  100],
        "price": [780, 690, 600, 540, 510, 500, 510, 560, 660, 770, 880, 990],
    }
    test = {"age": [3, 22, 45, 90]}

    df = pd.DataFrame(data)
    print("training data (price in $k):")
    print(df.to_string(index=False))

    poly = PolynomialFeatures(degree=2, include_bias=False)
    X = poly.fit_transform(df[["age"]].values)
    reg = LinearRegression().fit(X, df["price"])

    # PolynomialFeatures returns columns in order [x, x^2]
    b, a = reg.coef_
    c = reg.intercept_
    print(f"\nfit: price = {a:.4f} * age^2 + {b:.4f} * age + {c:.4f}")

    tst = pd.DataFrame(test)
    tst["price"] = reg.predict(poly.transform(tst[["age"]].values))
    print("\npredictions:")
    print(tst.to_string(index=False))

    age_grid = np.linspace(df.age.min(), df.age.max(), 200)
    curve = reg.predict(poly.transform(age_grid.reshape(-1, 1)))

    plt.figure()
    plt.xlabel("house age (yr)")
    plt.ylabel("price ($k)")
    plt.scatter(df.age, df.price, color="magenta", marker="*", s=80, label="training")
    plt.plot(age_grid, curve, color="blue", label="quadratic fit")
    plt.scatter(tst.age, tst.price, color="red", label="predicted")
    plt.legend()
    plt.savefig(PLOT_PATH, bbox_inches="tight")
    print(f"\nWrote {PLOT_PATH}")


if __name__ == "__main__":
    main()
