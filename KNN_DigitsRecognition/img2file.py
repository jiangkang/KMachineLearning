# encoding=utf-8

from PIL import Image

import numpy as np


def img2txt(img_path, txt_name):
    """
    将图像数据转换为txt文件
    :param img_path: 图像文件路径
    :type txt_name: 输出txt文件路径
    """
    im = Image.open(img_path).convert('1').resize((32, 32))  # type:Image.Image

    data = np.asarray(im)

    np.savetxt(txt_name, data, fmt='%d', delimiter='')

