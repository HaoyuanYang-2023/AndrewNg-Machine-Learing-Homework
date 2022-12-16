import numpy as np

a = np.array([[0, 1],
              [1, 4],
              [2, 7],
              [3, 10]])
print(a.mean(axis=0))
print(a.std(axis=0))
a = (a - a.mean(axis=0)) / a.std()
print(a)
print(a[0,0] * 3 + 1)