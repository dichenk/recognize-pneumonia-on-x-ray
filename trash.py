import glob
import numpy as np


img_path_train_begin = 'img/train/'
img_path_train_nrm_end = 'NORMAL_SMALL_half/*'
img_path_train_pn_end = 'PNEUMONIA_SMALL_half/*'
float_formatter = "{:.5f}".format  # формат вывода для действительных чисел
np.set_printoptions(formatter={'float_kind':float_formatter})  # формат вывода для действительных чисел
boarder = 85


img_train_normal_name = []
img_train_normal_np = []
a = 0
for i in glob.glob(img_path_train_begin + img_path_train_nrm_end):
    img_train_normal_name.append(i)
    if a < 10: print(i)
    a += 1