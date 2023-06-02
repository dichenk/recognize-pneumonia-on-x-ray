import random
import os
import numpy as np  # пакет для работы с массивами данных
import csv

my_list = None
with open('подонки_5.csv') as f:
    reader = csv.reader(f)
    my_list = list(reader)
    # print(my_list)
# print(len(my_list))

for i in range(0, len(my_list)):
    if random.randint(1,20) < 18:
        try:
            os.remove(my_list[i][0])
        except:
            pass
    

