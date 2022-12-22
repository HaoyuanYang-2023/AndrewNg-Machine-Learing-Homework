import numpy as np
import matplotlib.pyplot as plt


def visual_result(label_pred, x, centroids, fig_size, fig_dpi):
    """
    Visualization of clustering result
    :param label_pred: (n, )
    :param x:
    :param centroids:
    :param fig_size:
    :param fig_dpi:
    :return:
    """
    k = centroids.shape[0]
    plt.figure(figsize=fig_size, dpi=fig_dpi)
    for i in range(k):
        indices = np.where(label_pred == i)[0]
        cls_i = x[indices, :]
        plt.scatter(cls_i[:, 0], cls_i[:, 1], color=plt.cm.Set1(i % 8), s=10, label='class ' + str(i))
    plt.scatter(x=centroids[:, 0], y=centroids[:, 1], color=plt.cm.Set1(9), marker='*', label='Cluster center')
    plt.legend(loc=2)
    plt.title("Clustering Result")
    plt.show()


def euclid(p1, p2):
    """
    Compute Euclidian distance between p1 and p2
    :param p1: ndarray with shape of (n_samples, n_clusters, n_features)
    :param p2: ndarray with shape of (n_samples, n_clusters, n_features)
    :return: Euclidian distance. ndarray with shape (n_samples, n_clusters)
    """
    # (n,k,d)
    distance = (p1 - p2) ** 2
    # return (n,k)
    return np.sqrt(np.sum(distance, axis=2))


class KMeans:
    """
    KMeans Class
    ---------------------------------------------------------------
    Parameters:
    k:
        Number of clusters to classify.
    max_iter:
        Max iterations of k-means algorithm.

    ---------------------------------------------------------------
    Attributes:
    distance: ndarray with shape of (n_samples, n_clusters)
        The distance between each sample and each centroid.
    centroids: ndarray with shape od (n_clusters, n_features)
    """

    def __init__(self, k, max_iter=1000):
        self.k = k
        self.max_iter = max_iter

        self.distance = None
        self.centroids = None
        self.cls = None
        self.__stop = False

    def __init_centroids(self, x):
        """
        Uniform sampling on each dimension to init centroids
        :param x: (n,d)
        :return:
        """
        idx = np.sort(x, axis=0)
        _k = np.linspace(1, self.k, self.k)
        step = x.shape[0] / (self.k + 1)
        sample_idx = step * _k
        sample_idx = np.floor(sample_idx).astype(int)
        self.centroids = idx[sample_idx, :]

    def __get_distance(self, x):
        """
        x: (n,d) -> (n,k,d)
        centroids (k,d) -> (n,k,d)
        distance (n,k)

        :param x: (n,d)
        :return: (n,k)
        """
        _x = np.expand_dims(x, axis=1).repeat(repeats=self.k, axis=1)
        _c = np.expand_dims(self.centroids, axis=0).repeat(repeats=x.shape[0], axis=0)

        self.distance = euclid(_x, _c)

    def __update_centroids(self, x):
        """
        Compute the average value on each dimension of each sample to for each class to get new centroid
        If the bias of new centroid and current centroids is the same,
        new centroids can be seen as the result
        :param x: ndarray (n,d), samples
        :return:
        """
        new_centroids = self.centroids.copy()
        for i in range(self.k):
            indices = np.where(self.cls == i)[0]
            new_centroids[i, :] = np.average(x[indices, :], axis=0)
        if not np.equal(self.centroids, new_centroids).all():
            self.centroids = new_centroids
        else:
            self.__stop = True

    def __get_closest_centroid(self):
        """
        distance (n,k)
        :return: cls (n,)
        """
        self.cls = np.argmin(self.distance, axis=1)

    def fit(self, x):
        """

        :param x: ndarray (n,d)
        :return:
        """
        self.__init_centroids(x)
        for i in range(self.max_iter):
            self.__get_distance(x)
            self.__get_closest_centroid()
            if not self.__stop:
                self.__update_centroids(x)
            else:
                break
        if not self.__stop:
            print("KMeans fails to converge, please try larger max_inter.")
        return self.cls, self.centroids
