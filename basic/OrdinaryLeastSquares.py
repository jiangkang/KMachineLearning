# encoding=utf-8

import matplotlib.pyplot as plt
import numpy as np
from sklearn import datasets, linear_model

reg = linear_model.LinearRegression()

reg.fit(X=[[0, 0], [1, 1], [2, 2]], y=[0, 1, 2])



print(reg.coef_)
