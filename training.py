import numpy as np  # пакет для работы с массивами данных
from PIL import Image  # пакет для работы с изображениями
import random as rnd
import glob  # пакет для работы с файловой системой

img_path_train_begin = 'img/train/'
img_path_train_nrm_end = 'NORMAL_SMALL_half/*'
img_path_train_pn_end = 'PNEUMONIA_SMALL_half/*'
float_formatter = "{:.5f}".format  # формат вывода для действительных чисел
np.set_printoptions(formatter={'float_kind':float_formatter})  # формат вывода для действительных чисел
boarder = 125

#  чтение изображений
def read_img(our_img):  # картинка -> np.array
    our_img = Image.open(our_img)
    our_img = our_img.convert('L')  # в чб
    our_img = our_img.point(lambda p: 1 if p > boarder else 0)  # бинаризация
    our_img = np.array(our_img, dtype='float64')  # переформатируем изображение в массив
    our_img = our_img.ravel()  # make a 1-dimensional view of arr
    return our_img

img_train_normal_name = []
img_train_normal_np = []
for i in glob.glob(img_path_train_begin + img_path_train_nrm_end):
    img_train_normal_name.append(i)
    img_train_normal_np.append(read_img(i))

img_train_pn_name = []
img_train_pn_np = []
for i in glob.glob(img_path_train_begin + img_path_train_pn_end):
    img_train_pn_name.append(i)
    img_train_pn_np.append(read_img(i))


def neural_network(inp_pix, inp_wei):  # работа нейросети: картинка и веса
    pred = np.zeros(len(inp_wei[0]))
    for count, value in enumerate(np.transpose(inp_wei)):
        pred[count] = sum(inp_pix * value)
    return pred


def give_me_delta(img, dlt):  # получение коррекции веса (нужно для расчет весов)
    out = np.zeros((len(dlt), len(img)))
    out += img
    out = np.transpose(out)
    out *= dlt
    return out


keys = np.array([[1, 0], [0, 1]])  #normal and pneumonia
# weights = np.zeros((len(img_train_normal_np[0]), len(keys)), dtype='float64')  # веса
weights = np.loadtxt('text3.csv', delimiter=',')  # веса


alpha = 0.00001  # коэффициент
temp_var = 300000  # для обучения ставили 2000000
while temp_var > 0:
    temp_var -= 1
    # print(temp_var)
    n = rnd.randint(0, 1) # ставили вероятсноть 0, 3
    if n == 0:
        n_2 = rnd.randint(0, len(img_train_normal_np) - 1)
        reference_img = img_train_normal_np[n_2]
        key = keys[0]
    else:
        n_2 = rnd.randint(0, len(img_train_pn_np) - 1)
        reference_img = img_train_pn_np[n_2]
        key = keys[1]
    prediction = neural_network(reference_img, weights)  # 1 шаг. Отправляем изображение 1 в нейросеть
    if temp_var % 10000 == 0:
        print(f'Я сейчас на {temp_var} шаге, prediction: {prediction}')
    delta = prediction - key  # разница ожидаемого и полученного значения распознавания
    error = delta ** 2  # считаем ошибку
    weight_deltas = give_me_delta(reference_img, delta)  # то, на что мы будем корректировать вес
    weights -= alpha * weight_deltas

np.savetxt('text4.csv', weights, delimiter=',')