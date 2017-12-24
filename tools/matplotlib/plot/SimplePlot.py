# encoding=utf-8

import matplotlib.pyplot as plt
import numpy as np

# 要显示的数据
# x -> (0.0,2.0,step=0.01) y-> sin(2 * pi * x) + 1
x = np.arange(0.0, 2.0, step=0.01)
y = np.sin(2 * np.pi * x) + 1

# 相当于fig = plt.figure, ax = fig.add_subplot(111)
fig, ax = plt.subplots()

# 坐标轴描述
ax.set(xlabel='x', ylabel='y', title='y = sin(2πx) + 1')

# 网格
ax.grid()

# 映射
ax.plot(x, y)

# 保存图片
fig.savefig("../../../res/simple_plot.png")

# 显示
plt.show()
