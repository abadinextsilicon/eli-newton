from typing import List
import numpy as np
from matplotlib import pyplot as plt


def newton_dev_diff(nodes: List[float], images: List[float]):
    n = len(nodes)
    result = np.zeros((n, n))
    result[:, 0] = images
    for i in range(1, len(nodes)):
        for j in range(1, i + 1):
            result[i][j] = (result[i][j - 1] - result[i - 1][j - 1]) / (nodes[i] - nodes[i - j])
    diagonal = np.diagonal(result)
    return diagonal, result


def newton_poly(x, nodes):
    y = []
    y.append(1)
    for i in range(1, len(nodes) + 1):
        y.append(y[i - 1] * (x - nodes[i - 1]))
    return y


def PolyCoefficients(x, coeffs, nodes):
    o = len(coeffs)
    n = newton_poly(x, nodes)
    y = 0
    for i in range(o):
        y += coeffs[i] * n[i]
    return y


def getInput():
    result = []
    for n in (5, 10, 15, 20):
        nodes = []
        images = []
        for i in range(1, n + 1):
            x = -5 + 10 * ((i - 1) / (n - 1))
            nodes.append(x)
            images.append(1 / (1 + x * x))
        result.append((nodes, images))
    return result


# Press the green button in the gutter to run the script.


def org_func(x):
    y = 1 / (1 + x * x)
    return y


if __name__ == '__main__':

    # diagonal, result = newton_dev_diff([0, 1, 3, 4, 7], [1, 3, 49, 129, 813])

    # print(result);
    r = getInput()
    x = np.linspace(-4, 4, 1000)
    plt.xlim([-4, 4])
    plt.ylim([0, 8])
    for p1, im1 in r:
        diagonal, result = newton_dev_diff(p1, im1)
        plt.plot(x, PolyCoefficients(x, diagonal, p1))

    plt.plot(x, org_func(x))
    plt.show()
