import matplotlib.pyplot as plt
import numpy as np


def gaussian_kernel(x1, x2, sigma):
    """
    Gaussian Kernel function
    :param x1: input (1, n)
    :param x2: input (1, n)
    :param sigma: sigma
    :return: gaussian value
    """
    return np.exp(-np.sum(x1 - x2) ** 2 / (2 * (sigma ** 2)))


def show_boundary(svc, scale, fig_size, fig_dpi, positive_data, negative_data, term):
    """
    Show SVM classification boundary plot
    :param svc: instance of SVC, fitted and probability=True
    :param scale: scale for x-axis and y-axis
    :param fig_size: figure size, tuple (w, h)
    :param fig_dpi: figure dpi, int
    :param positive_data: positive data for dataset (n, d)
    :param negative_data: negative data for dataset (n, d)
    :param term: width for classification boundary
    :return: decision plot
    """
    t1 = np.linspace(scale[0, 0], scale[0, 1], 500)
    t2 = np.linspace(scale[1, 0], scale[1, 1], 500)
    coordinates = np.array([[x, y] for x in t1 for y in t2])
    prob = svc.predict_proba(coordinates)
    idx1 = np.where(np.logical_and(prob[:, 1] > 0.5 - term, prob[:, 1] < 0.5 + term))[0]
    my_bd = coordinates[idx1]
    plt.figure(figsize=fig_size, dpi=fig_dpi)
    plt.scatter(x=my_bd[:, 0], y=my_bd[:, 1], s=10, color="yellow", label="My Decision Boundary")
    plt.scatter(x=positive_data[:, 0], y=positive_data[:, 1], s=10, color="red", label="positive")
    plt.scatter(x=negative_data[:, 0], y=negative_data[:, 1], s=10, label="negative")
    plt.title('Decision Boundary')
    plt.legend(loc=2)
    plt.show()
