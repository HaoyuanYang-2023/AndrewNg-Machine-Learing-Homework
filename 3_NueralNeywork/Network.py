import numpy as np
from LogisticRegression.LogisticRegression import sigmoid


def onehot_encode(label):
    """
    Onehot编码
    :param label: (n,)
    :return: onehot label (n, n_cls)
    """
    # cls (n_cls,)
    cls = np.unique(label)
    y_matrix = []
    for cls_idx in cls:
        y_matrix.append((label == cls_idx).astype(int))
    one_hot = np.array(y_matrix).T
    return one_hot, cls


class ForwardModel:
    def __init__(self):
        self.theta1 = None
        self.theta2 = None

    def load_parameters(self, parameters):
        self.theta1 = parameters[0]
        self.theta2 = parameters[1]

    def __call__(self, x, *args, **kwargs):
        # x (n,d)
        t = np.ones(shape=(x.shape[0], 1))
        # x (n,d+1)
        a1 = np.concatenate((t, x), axis=1)
        # a2 （n, hidden_size）
        a2 = sigmoid(np.matmul(a1, self.theta1))
        # a2 （n, hidden_size + 1）
        a2 = np.concatenate((t, a2), axis=1)
        # a3 （n, cls_n）
        a3 = sigmoid(np.matmul(a2, self.theta2))
        return a3



