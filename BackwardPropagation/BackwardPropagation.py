import numpy as np
from LogisticRegression.LogisticRegression import sigmoid


def g_sigmoid(x):
    """
    Derivative of sigmoid function
            g'(x) = g(x)(1 - g(x)),
    where g() is the sigmoid function
    :param x: input with any shape
    :return: Derivative value of sigmoid
    """
    return np.multiply(sigmoid(x), (1 - sigmoid(x)))


def loss(pred, target):
    """
    Loss function
            Loss = -target * log(pred) - (1 - target) * log(1-pred),
    where target in a one-hot vector
    :param pred: predicted distribution with shape of (n, 10)
    :param target: target distribution with shape of (n, 10)
    :return: loss
    """
    return np.sum(-np.multiply(target, np.log(pred)) - np.multiply((1 - target), np.log(1 - pred)))


def gradient(model, output, target):
    """
    Get gradient of model
    :param model: NN model
    :param output: shape of (n, 10)
    :param target: shape of (n, 10)
    :return: gradient with shape of parameters' shape
    """
    # theta1 (401, 25); theta2 (26, 10)
    n = output.shape[0]
    # d3 (n, 10)
    d3 = output - target
    # t (n, 1)
    t = np.ones(shape=(n, 1))
    # z2 (n, 25); z2_ (n, 26)
    z2_ = np.concatenate((t, model.z2), axis=1)
    # g_prime_z2 (n, 26)
    g_prime_z2 = g_sigmoid(z2_)
    # d3 @ model.theta2.T (n, 26)
    # skip d2_0, d2 (n, 25)
    d2 = np.multiply(d3 @ model.theta2.T, g_prime_z2)[:, 1:]
    # (n, 26).T @ (n, 10) = (26, 10)
    delta2 = model.a2.T @ d3
    # (n, 401).T @ (n, 25) = (401, 25)
    delta1 = model.a1.T @ d2

    return delta1 / n, delta2 / n


def regularized_gradient(model, output, target, scale):
    """
    Get regularized("L2") gradient of model
    Don't regularize the bias term of parameters
    :param model: NN model
    :param output: Output of model with shape of (n, 10)
    :param target: target distribution with shape of (n, 10)
    :param scale: scale for regularization
    :return: regularized gradient with shape of parameters' shape
    """
    # delta1 (401, 25); delta2 (26, 10)
    delta1, delta2 = gradient(model=model, output=output, target=target)
    n = output.shape[0]

    theta1 = model.theta1
    theta1[0, :] = 0
    reg_term_d1 = (scale / n) * theta1
    delta1 = delta1 + reg_term_d1

    theta2 = model.theta2
    theta2[0, :] = 0
    reg_term_d2 = (scale / n) * theta2
    delta2 = delta2 + reg_term_d2

    return delta1, delta2


def regularized_loss(pred, target, theta, scale):
    """
    Regularized loss function
            Regularized_loss = Loss + lambda / (2 * m) * (square_sum(theta1) + square_sum(theta2)),
    where theta1 and theta2 exclude the bias term
    :param pred: predicted distribution with shape of (n, 1)
    :param target: target distribution with shape of (n, 1)
    :param theta: parameters list of model with shape of (2, parameter_shape)
    :param scale: scale for regularize term
    :return: regularized loss
    """
    m = pred.shape[0]
    # theta1 (401, 25)
    theta1 = theta[0]
    # theta2 (26, 10)
    theta2 = theta[1]
    reg_1 = scale / (2 * m) * (theta1[1:, :] ** 2).sum()
    reg_2 = scale / (2 * m) * (theta2[1:, :] ** 2).sum()
    row_loss = loss(pred=pred, target=target) / m
    return row_loss + reg_1 + reg_2


class BackPropModel:
    """
    BackPropagation Model
    parameter shape: (401, 25) and (26, 10)
    """

    def __init__(self, penalty="L2", scale=0):
        """
        Initialize Function
        :param penalty: Regularization
        :param scale: Lambda for regularization
        """
        self.theta1 = None
        self.theta2 = None
        self.penalty = penalty
        self.scale = scale
        self.a1 = None
        self.z2 = None
        self.a2 = None
        self.z3 = None

    def load_parameters(self, parameters):
        """
        Load parameters
        :param parameters: shape of [2, parameters_shape]
        :return:
        """
        self.theta1 = parameters[0]
        self.theta2 = parameters[1]

    def init_parameters(self, l_in, l_out):
        """
        Initialize parameters with uniform distribution in [-epsilon, epsilon]
                        epsilon = np.sqrt(6 / (l_in+l_out))

        :param l_in: Number of unit of input layer
        :param l_out: Number of unit of output layer
        :return:
        """
        epsilon = np.sqrt(6 / (l_in + l_out))
        parameters = np.random.uniform(low=-epsilon, high=epsilon, size=401 * 25 + 26 * 10)
        self.theta1 = parameters[:401 * 25].reshape(401, 25)
        self.theta2 = parameters[401 * 25:].reshape(26, 10)

    def optimize(self, g, lr=0.01):
        """
        Optimize parameters via Batch Gradient Descent
                theta = theta - alpha * gradient

        :param lr: learning rate with default 0.01
        :param g: gradient list for parameters with shape of [2, parameters_shape]
        :return:
        """
        # gradient[0] (401, 25); gradient[1] (26, 10)
        self.theta1 = self.theta1 - lr * g[0]
        self.theta2 = self.theta2 - lr * g[1]

    def __call__(self, x, *args, **kwargs):
        """
        Forward Propagation
        :param x: Input of model with shape of (n, 400)
        :return: Calculation result for each layer
        """
        # x (n, 400)
        t = np.ones(shape=(x.shape[0], 1))
        # x (n, 401)
        self.a1 = np.concatenate((t, x), axis=1)

        # z2 (n, 25); a2 (n, 25)
        self.z2 = np.matmul(self.a1, self.theta1)
        a2 = sigmoid(self.z2)

        # a2 (n, 26）
        self.a2 = np.concatenate((t, a2), axis=1)

        # z3 (n, 10）
        self.z3 = np.matmul(self.a2, self.theta2)
        # a3 (n, 10)
        a3 = sigmoid(self.z3)
        return a3
