import numpy as np

from LinearRegression.utils import LinearRegression
train_x_ex = np.zeros([77,1])
train_y_ex = np.ones([77,1])
linear_reg = LinearRegression(train_x_ex,train_y_ex)
t,l=linear_reg.run()
print(t)