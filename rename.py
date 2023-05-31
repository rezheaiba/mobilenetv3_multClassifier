"""
# @Time    : 2022/8/3 14:12
# @File    : 批量改名.py
# @Author  : rezheaiba
"""
import os
import shutil
import random

import cv2
import numpy as np
from PIL import Image

root = '../datasets/final-8_mult/train'

filelist = os.listdir(root)
print(filelist)
for path in filelist:
    sortId = 1
    current_path = os.path.join(root, path)
    currentlist = os.listdir(current_path)
    for name in currentlist:
        os.rename(os.path.join(current_path, name), os.path.join(current_path, (path + '_' +str('%06d' % sortId) + '.jpg')))
        sortId += 1