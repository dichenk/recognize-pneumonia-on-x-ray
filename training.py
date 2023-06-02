import numpy as np  # пакет для работы с массивами данных
from PIL import Image  # пакет для работы с изображениями
import random as rnd  # пакет для генерации случайных значений
import glob  # пакет для работы с файловой системой



def read_img(our_img):  # функция чтения изображений и преобразования в массив np
    our_img = Image.open(our_img)
    our_img = our_img.convert('L')  # в чб
    our_img = our_img.point(lambda p: 1 if p > boarder else 0)  # бинаризация
    our_img = np.array(our_img, dtype='float64')  # переформатируем изображение в массив
    our_img = our_img.ravel()  # формируем одномерный массив
    return our_img

def neural_network(inp_pix, inp_wei):  # работа нейросети: картинки и веса
    pred = np.zeros(len(inp_wei[0]))
    for count, value in enumerate(np.transpose(inp_wei)):
        pred[count] = sum(inp_pix * value)
    return pred

def give_me_delta(img, dlt):  # получение коррекции веса (нужно для расчета весов)
    out = np.zeros((len(dlt), len(img)))
    out += img
    out = np.transpose(out)
    out *= dlt
    return out


img_path_train_nrm = 'img/train/NORMAL_SMALL_half/*'
img_path_train_pn = 'img/train/PNEUMONIA_SMALL_half/*'
float_formatter = "{:.5f}".format  # формат вывода для действительных чисел
np.set_printoptions(formatter={'float_kind':float_formatter})  # формат вывода для действительных чисел
boarder = 85  # граница бинаризации
img_train_normal_np = []  # массив изображений здоровых пациентов
img_train_pn_np = []  # массив изображений больных пациентов
keys = np.array([[1, 0], [0, 1]])  # шаблоны для обучения
alpha = 0.00001  # коэффициент
temp_var = 0  # количество циклов обучения.

for i in glob.glob(img_path_train_nrm):  # чтение изображений здоровых пациентов
    img_train_normal_np.append(read_img(i))

for i in glob.glob(img_path_train_pn):  # чтение изображений больных пациентов
    img_train_pn_np.append(read_img(i))

weights = np.zeros((len(img_train_normal_np[0]), len(keys)), dtype='float64')  # начальный вес

while temp_var > 0:  # цикл обучения
    temp_var -= 1  # счетчик циклов обучения
    n = rnd.randint(0, 1)  # рандомизация обучения

    if n == 0:  # обучение на рентгене здорового ребенка
        n_2 = rnd.randint(0, len(img_train_normal_np) - 1)
        reference_img = img_train_normal_np[n_2]
        key = keys[0]
    else:  # обучение на рентгене больного ребенка
        n_2 = rnd.randint(0, len(img_train_pn_np) - 1)
        reference_img = img_train_pn_np[n_2]
        key = keys[1]

    prediction = neural_network(reference_img, weights)  # 1 шаг. Отправляем изображение 1 в нейросеть
    delta = prediction - key  # разница ожидаемого и полученного значения распознавания
    error = delta ** 2  # считаем ошибку
    weight_deltas = give_me_delta(reference_img, delta)  # то, на что мы будем корректировать вес
    weights -= alpha * weight_deltas  # коррекция веса

# np.savetxt('weights.csv', weights, delimiter=',')  # сохранение результата работы обучения в формате csv