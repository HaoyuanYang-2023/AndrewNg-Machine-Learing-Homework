import numpy as np


def loss(pred, target):
    """
    MSE Loss
    :param pred: ndarray, (n_sample, n_feature), prediction
    :param target: ndarray, (n_sample, n_feature), target
    :return:
    """
    return np.sum(np.power((pred - target), 2)) / (2 * pred.shape[0])


def reg_loss(pred, target, theta, scale):
    """
    Regularized Loss (L2 regularize)
    :param pred:
    :param target:
    :param theta: parameters
    :param scale: lambda for regularize
    :return:
    """
    reg_term = theta.copy()  # (1, 2)
    reg_term[:, 0] = 0
    return loss(pred, target) + scale / (2 * pred.shape[0]) * reg_term.sum()


def gradient(x, pred, target):
    """
    Get gradient for model
    :param x:
    :param pred:
    :param target:
    :return:
    """
    # x(n,2)
    t = np.ones(shape=(x.shape[0], 1))
    x = np.concatenate((t, x), axis=1)
    # pred-target (n,1)
    # gradient (1,2)
    g = np.matmul((pred-target).T, x)
    return g / pred.shape[0]


def reg_gradient(x, pred, target, theta, scale):
    """
    Regularized gradient
    :param x:
    :param pred:
    :param target:
    :param theta:
    :param scale:
    :return:
    """
    g = gradient(x, pred, target)
    reg_term = theta.copy()
    reg_term[:, 0] = 0
    return g + (scale / pred.shape[0]) * reg_term


class LinearRegression:
    def __init__(self):
        self.theta = None

    def init_theta(self, shape):
        self.theta = np.zeros(shape)
        # self.theta = np.ones(shape)

    def optimize(self, g, lr):
        self.theta = self.theta - lr * g

    def load_parameters(self, parameters):
        self.theta = parameters

    def __call__(self, x, *args, **kwargs):
        #  x (n,2)
        t = np.ones(shape=(x.shape[0], 1))
        x = np.concatenate((t, x), axis=1)
        # (n,2)@(2,1)=(n,1)
        return x @ self.theta.T
