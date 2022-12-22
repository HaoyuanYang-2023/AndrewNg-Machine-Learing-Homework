import numpy as np


class PCA:
    """
    ----------------------------------------------------------------------------
    Attributes:
    components_: ndarray with shape of (n_features, n_components)
        Principal axes in feature space, representing the directions of maximum
        variance in the data.

    explained_variance_ : ndarray of shape (n_components,)
        The amount of variance explained by each of the selected components.

    explained_variance_ratio_ : ndarray of shape (n_components,)
        Percentage of variance explained by each of the selected components.
    """
    def __init__(self, n_components):
        self.n_components = n_components
        self.explained_variance_ = None
        self.explained_variance_ratio_ = None
        self.components_ = None

        self.__mean = None

    def fit(self, x):
        """
        :param x: (n,d)
        :return:
        """
        self.__mean = x.mean(axis=0)
        x_norm = x - self.__mean
        x_cov = (x_norm.T @ x_norm) / (x.shape[0] - 1)
        vectors, variance, _ = np.linalg.svd(x_cov)
        # (n_feature, n_components)
        self.components_ = vectors[:, :self.n_components]
        if len(self.components_.shape) == 1:
            self.components_ = np.expand_dims(vectors[:, :self.n_components], axis=1)
        self.explained_variance_ = variance[:self.n_components]
        self.explained_variance_ratio_ = self.explained_variance_ / variance.sum()

    def transform(self, x):
        """
        :param x: (n, n_feature)
        :return:
        """
        if self.__mean is not None:
            x = x - self.__mean
        x_transformed = x @ self.components_
        return x_transformed
