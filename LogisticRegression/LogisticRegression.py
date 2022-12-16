import numpy as np


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def bce_loss(pred, target):
    """
    计算误差
    :param pred: 预测
    :param target: ground truth
    :return: 损失序列
    """
    return np.mean(-target * np.log(pred))


class LogisticRegression:
    """
    Logistic回归类
    """

    def __init__(self, x, y, val_x, val_y, epoch=100, lr=0.1, normalize=True, regularize=None, scale=0):
        """
        初始化
        :param x: 样本, (sample_number, dimension)
        :param y: 标签, (sample_numer, 1)
        :param epoch: 训练迭代次数
        :param lr: 学习率
        """
        self.theta = None
        self.loss = []
        self.val_loss = []
        self.n = x.shape[0]
        self.d = x.shape[1]

        self.epoch = epoch
        self.lr = lr

        t = np.ones(shape=(self.n, 1))

        self.normalize = normalize

        if self.normalize:
            self.x_std = x.std(axis=0)
            self.x_mean = x.mean(axis=0)
            self.y_mean = y.mean(axis=0)
            self.y_std = y.std(axis=0)
            x = (x - self.x_mean) / self.x_std

        self.y = y
        self.x = np.concatenate((t, x), axis=1)

        # self.val_x = (val_x - val_x.mean(axis=0)) / val_x.std(axis=0)
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

    def get_loss(self, pred, target):
        """
        计算损失
        :param pred: 预测 (n,1)
        :param target: ground truth
        """
        loss = bce_loss(pred, target)
        self.loss.append(loss)
        return loss

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
        # term (d+1,)
        term = np.expand_dims(term.diagonal().copy().T, axis=0)

        if self.regularize == "L2":
            re = self.scale / self.n * self.theta[0, 1:]
            re = np.expand_dims(np.array(re), axis=0)
            re = np.concatenate((np.array([[0]]), re), axis=1)
            # re [0,...] (1,d+1)
            self.theta = self.theta - self.lr * (term / self.n + re)
        # update parameters
        else:
            self.theta = self.theta - self.lr * (term / self.n)

    def validation(self, x, y):
        x = (x - x.mean(axis=0)) / x.std(axis=0)
        outputs = self.get_prob(x)
        self.val_loss.append(bce_loss(outputs, y))
        predicted = np.expand_dims(np.where(outputs[:, 0] > 0.5, 1, 0), axis=1)
        count = np.sum(predicted == y)
        print("Accuracy on Val set: {:.2f}%".format(count / y.shape[0] * 100))

    def test(self, x, y):
        outputs = self.get_prob(x)
        predicted = np.expand_dims(np.where(outputs[:, 0] > 0.5, 1, 0), axis=1)
        count = np.sum(predicted == y)
        # print("Accuracy on Test set: {:.2f}%".format(count / y.shape[0] * 100))
        return count / y.shape[0], bce_loss(outputs, y)

    def train(self):
        """
        训练Logistic回归
        :return: 参数矩阵theta (1,d+1); 损失序列 loss
        """
        self.init_theta()

        for i in range(self.epoch):
            # pred (1,n); theta (1,d+1); self.x.T (d+1, n)
            z = np.matmul(self.theta, self.x.T).T
            # pred (n,1)
            pred = sigmoid(z)
            curr_loss = self.get_loss(pred, self.y)
            if self.regularize == "L2":
                curr_loss += self.scale / self.n * np.sum(self.theta[0, 1:] ** 2)

            self.gradient_decent(pred)

            print("Epoch: {}/{}, Train Loss: {:.4f}".format(i + 1, self.epoch, curr_loss))
            self.validation(self.val_x, self.val_y)

        if self.normalize:
            y_mean = np.mean(z, axis=0)
            self.theta[0, 1:] = self.theta[0, 1:] / self.x_std.T
            self.theta[0, 0] = y_mean - np.dot(self.theta[0, 1:], self.x_mean.T)
        return self.theta, self.loss, self.val_loss

    def get_prob(self, x):
        """
        回归预测
        :param x: 输入样本 (n,d)
        :return: 预测结果 (n,1)
        """
        # (d,1)
        # x = (x - x.mean(axis=0)) / x.std(axis=0)
        t = np.ones(shape=(x.shape[0], 1))
        x = np.concatenate((t, x), axis=1)
        pred = sigmoid(np.matmul(self.theta, x.T))
        return pred.T

    def get_inner_product(self, x):
        t = np.ones(shape=(x.shape[0], 1))
        x = np.concatenate((t, x), axis=1)
        return np.matmul(self.theta, x.T)

    def predict(self, x):
        prob = self.get_prob(x)
        return np.expand_dims(np.where(prob[:, 0] > 0.5, 1, 0), axis=1)