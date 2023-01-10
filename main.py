import numpy as np
import matplotlib.pyplot as plt


def a(p, k):
    return -np.log((1 - (1 - p) / (k - 1)) ** (k - 1))


def get_cross_entropy_loss_append_item_fig():
    x = np.arange(0.8, 0.9, 0.001)
    k3 = a(x, 3)
    k4 = a(x, 4)
    k5 = a(x, 5)
    fig, ax = plt.subplots(figsize=(8, 6), dpi=100)
    ax.plot(x, -np.log(x), label="-log(x)", linestyle='--')
    ax.plot(x, k3, color="red", label="k=3")
    ax.plot(x, k4, color="blue", label="k=4")
    ax.plot(x, k5, color="black", label="k=5")
    ax.set_title("Append item function line")
    ax.set_xlabel("p")
    plt.legend(loc=1)
    plt.savefig("./imgs/Wu_MultiClass_Loss")

get_cross_entropy_loss_append_item_fig()