import numpy as np


def square_loss(pred, target):
    """
    计算平方误差
    :param pred: 预测
    :param target: ground truth
    :return: 损失序列
    """
    return np.sum(np.power((pred - target), 2))


def compute_loss(pred, target):
    """
    计算归一化平均损失
    :param pred: 预测
    :param target: ground truth
    :return: 损失
    """
    pred = (pred - pred.mean(axis=0)) / pred.std(axis=0)
    target = (pred - target.mean(axis=0)) / target.std(axis=0)
    loss = square_loss(pred, target)
    return np.sum(loss) / (2 * pred.shape[0])


class LinearRegression:
    """
    线性回归类
    """

    def __init__(self, x, y, val_x, val_y, epoch=100, lr=0.1, regularize=False, scale=0):
        """
        初始化
        :param x: 样本, (sample_number, dimension)
        :param y: 标签, (sample_numer, 1)
        :param epoch: 训练迭代次数
        :param lr: 学习率
        :param regularize: Regularize or not
        """
        self.theta = None
        self.loss = []
        self.val_loss = []
        self.n = x.shape[0]
        self.d = x.shape[1]

        self.epoch = epoch
        self.lr = lr

        t = np.ones(shape=(self.n, 1))

        self.x_std = x.std(axis=0)
        self.x_mean = x.mean(axis=0)
        self.y_mean = y.mean(axis=0)
        self.y_std = y.std(axis=0)

        x_norm = (x - self.x_mean) / self.x_std
        y_norm = (y - self.y_mean) / self.y_std

        self.y = y_norm
        self.x = np.concatenate((t, x_norm), axis=1)

        self.val_x = val_x
        self.val_y = val_y

        self.regularize = regularize
        self.scale = scale

    def init_theta(self):
        """
        初始化参数
        :return: theta (1, d+1)
        """
        self.theta = np.zeros(shape=(1, self.d + 1))

    def validation(self, x, y):
        x = (x - x.mean(axis=0)) / x.std(axis=0)
        y = (y - y.mean(axis=0)) / y.std(axis=0)
        outputs = self.predict(x)
        curr_loss = square_loss(outputs, y) / (2 * y.shape[0])
        return curr_loss

    def gradient_decent(self, pred):
        """
        实现梯度下降求解
        """
        # error (n,1)
        error = pred - self.y
        # term (d+1, 1)
        gradient = np.matmul(self.x.T, error)
        # term (1,d+1)
        gradient = gradient.T / pred.shape[0]
        if self.regularize:
            # reg_term (1, d+1)
            reg_term = self.scale / self.n * self.theta[:, 1:]
            reg_term[:, 0] = 0
            gradient = gradient + reg_term
        # update parameters
        self.theta = self.theta - (self.lr / self.n) * gradient

    def train(self):
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
            curr_loss = square_loss(pred, self.y) / (2 * self.n)
            val_loss = self.validation(self.val_x, self.val_y)
            self.gradient_decent(pred)
            # if self.regularize:
                # curr_loss = curr_loss + (self.scale / (2 * self.n)) * np.power(self.theta[:, 1:], 2).sum()
                # val_loss = val_loss + (self.scale / (2 * self.n)) * np.power(self.theta[:, 1:], 2).sum()
            self.val_loss.append(val_loss)
            self.loss.append(curr_loss)
            print("Epoch: {}/{}\tTrain Loss: {:.4f}\tVal loss: {:.4f}".format(i + 1, self.epoch, curr_loss, val_loss))

        # un_scaling parameters
        self.theta[0, 1:] = self.theta[0, 1:] / self.x_std.T * self.y_std[0]
        self.theta[0, 0] = self.theta[0, 0] * self.y_std[0] + self.y_mean[0] - np.dot(self.theta[0, 1:], self.x_mean.T)
        return self.theta, self.loss, self.val_loss

    def predict(self, x):
        """
        回归预测
        :param x: 输入样本 (n,d)
        :return: 预测结果 (n,1)
        """
        # (d,1)
        t = np.ones(shape=(x.shape[0], 1))
        x = np.concatenate((t, x), axis=1)
        pred = np.matmul(self.theta, x.T)
        return pred.T
