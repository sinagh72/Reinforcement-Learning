import math
import numpy as np


def equation(x : float, y: float):
    return (1 - math.pow(x, 2)) * y + (1 - math.pow(y, 2)) * x


if __name__ == "__main__":
    y_probs = [1 / 5, 3 / 5, 1 / 5]
    x_probs = [1 / 6, 3 / 6, 2 / 6]

    values = [-1, 0, 1]

    x_w = np.random.choice(values, 1, replace=False, p=x_probs)
    x = x_w[0]
    print("x init:", x)
    for t in range(1000):
        x_pre = x
        y = np.random.choice(values, 1, replace=False, p=y_probs)[0]
        x = equation(x, y)
        ex = np.sum(x*x_w)

        print(f"t:{t}, E[x_t+1]:{ex}, x_t+1: {x}, x_t:{x_pre}, y_t: {y}")

