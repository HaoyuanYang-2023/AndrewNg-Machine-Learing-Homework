import numpy as np


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
