import numpy as np


def square_loss(pred, target):
    """
    计算损失
    :param pred: 预测
    :param target: ground truth
    :return: 损失序列
    """
    return np.power((pred - target), 2)


class LinearRegression:
    """
    线性回归类
    """

    def __init__(self, x, y, epoch=100, lr=0.1):
        """
        初始化
        :param x: 样本, (sample_number, dimension)
        :param y: 标签, (sample_numer, 1)
        :param epoch: 训练迭代次数
        :param lr: 学习率
        """
        self.theta = None
        self.loss = []
        self.n = x.shape[0]
        self.d = x.shape[1]
        # self.y (n,1)
        self.epoch = epoch
        self.lr = lr

        t = np.ones(shape=(self.n, 1))
        # self.x (n, d+1)
        # self.std = y.std(axis=0)
        # self.mean = y.mean(axis=0)
        # x_norm = (x - x.mean(axis=0)) / x.std(axis=0)
        y_norm = (y - y.mean(axis=0)) / y.std(axis=0)
        self.y = y
        self.x = np.concatenate((t, x), axis=1)

    def init_theta(self):
        """
        初始化参数
        :return: theta (1, d+1)
        """
        self.theta = np.zeros(shape=(1, self.d + 1))

    def get_loss(self, pred, target):
        """
        计算损失
        :param pred: 预测 (n,1)
        :param target: ground truth
        """
        inner = square_loss(pred, target)
        self.loss.append(np.sum(inner) / (2 * self.n))
        return np.sum(inner) / (2 * self.n)

    def gradient_decent(self, pred):
        """
        实现梯度下降求解
        """
        # error (n,1)
        error = pred - self.y
        # error (d+1,n)
        error = error.T.repeat(repeats=self.d + 1, axis=0)
        # temp (1, d+1)
        temp = np.zeros_like(self.theta)
        # term (d+1,d+1); error (d+1, n); self.x (n, d+1)
        term = np.matmul(error, self.x)
        # term (1,d+1)
        term = term.diagonal().T
        # update parameters
        self.theta = self.theta - (self.lr / self.n) * term

    def run(self):
        """
        训练线性回归
        :return: 参数矩阵theta (1,d+1); 损失序列 loss
        """
        self.init_theta()

        for i in range(self.epoch):
            # pred (1,n); theta (1,d+1); self.x.T (d+1, n)
            pred = np.matmul(self.theta, self.x.T)
            # pred (n,1)
            pred = pred.T
            curr_loss = self.get_loss(pred, self.y)

            self.gradient_decent(pred)

            print("Epoch: {}/{}, Train Loss: {:.4f}".format(i + 1, self.epoch, curr_loss))
        # self.theta = self.theta * self.std.T + self.mean.T
        return self.theta, self.loss

    def prediction(self, x):
        """
        回归预测
        :param x: 输入样本 (n,d)
        :return: 预测结果 (n,1)
        """
        # (d,1)
        # if x.shape[0] > 1:
            # std = x.std(axis=0)
            # mean = x.mean(axis=0)
            # x = (x - mean) / std
        t = np.ones(shape=(x.shape[0], 1))
        x = np.concatenate((t, x), axis=1)
        pred = np.matmul(self.theta, x.T)
        return pred.T
        # return pred.T * self.std + self.mean
