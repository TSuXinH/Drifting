import cv2
import numpy as np
import matplotlib.pyplot as plt

# the correlation
y1 = np.random.rand(2000) * 2 - 1
eps = np.random.normal(0, .1, size=(2000, ))
y2 = - y1 + eps

plt.scatter(y1, y2, c='purple', s=2)
plt.show()

plt.hist(y1, bins=200)
plt.hist(y1+y2, bins=50)
plt.show()


x = np.random.randn(20000)
x1 = x[5000: 15000]
x2 = x[5001: 15001]
result = np.sum(x1 @ x1)
