#!/usr/bin/env python3.11

from helper import save_data, load_data

import cv2 as cv
from os.path import join

path = join("real_data", "grandpa")

pic = load_data(join(path, 'pic'))

#pic = cv.cvtColor(pic, cv.COLOR_BGR2RGB)
pic = pic[:,:,::-1]

cv.imwrite(join(path, 'quantized.png'), pic)
