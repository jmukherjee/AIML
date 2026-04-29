"""ML: Linear Regression - Multi Variate (CLI port of ../jupyter/01b_multi_var.ipynb).

Predict house price from house area (sq.ft.) and age (years).

    y = m1*x1 + m2*x2 + b

Run:
    python 01b_multi_var.py
"""

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn import linear_model

PLOT_PATH = "01b_multi_var_plot.png"


def main() -> None:
    data = {
        "area":  [2600, 3000, 3200, 3600, 4000, 4100],
        "age":   [20, 15, 18, 30, 8, 8],
        "price": [550000, 565000, 610000, 680000, 725000, 810000],
    }
    test = {
        "area": [2000, 3300, 4400],
        "age":  [10, 13, 16],
    }

    df = pd.DataFrame(data)
    print("training data:")
    print(df.to_string(index=False))

    reg = linear_model.LinearRegression()
    reg.fit(df[["area", "age"]], df["price"])

    print(f"\ncoef (area):   {reg.coef_[0]}")
    print(f"coef (age):    {reg.coef_[1]}")
    print(f"intercept:     {reg.intercept_}")

    tst = pd.DataFrame(test)
    tst["price"] = reg.predict(tst[["area", "age"]])
    print("\npredictions:")
    print(tst.to_string(index=False))

    # Regression-plane mesh over the training-data range.
    area_grid = np.linspace(df.area.min(), df.area.max(), 12)
    age_grid = np.linspace(df.age.min(), df.age.max(), 12)
    A, G = np.meshgrid(area_grid, age_grid)
    mesh = pd.DataFrame({"area": A.ravel(), "age": G.ravel()})
    Z = reg.predict(mesh).reshape(A.shape)

    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection="3d")
    ax.scatter(df.area, df.age, df.price, color="magenta", marker="*", label="training")
    ax.scatter(tst.area, tst.age, tst.price, color="red", label="predicted")
    ax.plot_surface(A, G, Z, color="blue", alpha=0.3)
    ax.set_xlabel("area")
    ax.set_ylabel("age")
    ax.set_zlabel("price")
    ax.set_title("Multivariate Linear Regression in 3D")
    ax.view_init(elev=20, azim=30)
    ax.legend()
    fig.savefig(PLOT_PATH, bbox_inches="tight")
    print(f"\nWrote {PLOT_PATH}")


if __name__ == "__main__":
    main()
