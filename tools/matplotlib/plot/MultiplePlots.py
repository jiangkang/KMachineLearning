# encoding=utf-8

import matplotlib.pyplot as plt
import numpy as np

x = np.linspace(-1, 1, num=50)

y1 = np.sin(np.pi * x * 2)
y2 = np.cos(np.pi * x * 2)
y3 = 1 / (1 + np.exp(-x))
y4 = np.tanh(np.pi * x * 2)

plt.subplots(2, 2)

# y1
plt.subplot(2, 2, 1)
plt.plot(x, y1, 'o-')
plt.xlabel('x')
plt.ylabel('y1')
plt.title('y1 = sin(2πx)')

# y2
plt.subplot(2, 2, 2)
plt.plot(x, y2, 'o-')
plt.xlabel('x')
plt.ylabel('y2')
plt.title('y2 = cos(2πx)')

plt.subplot(2, 2, 3)
plt.plot(x, y3, 'o-')
plt.xlabel('x')
plt.ylabel('y3')
plt.title('y3 = 1 / (1 + exp(-x))')

plt.subplot(2, 2, 4)
plt.plot(x, y4, 'o-')
plt.xlabel('x')
plt.ylabel('y4')
plt.title('y4 = tanh(x)')

plt.show()
