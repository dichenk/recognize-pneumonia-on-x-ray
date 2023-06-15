import training  # функции из файла training.py
import numpy as np  # пакет для работы с массивами данных
import glob  # пакет для работы с файловой системой


weight_test = np.loadtxt('weights.csv', delimiter=',')  # весовые коэффициенты
img_path_test_nrm = 'img/test/NORMAL_SMALL_half/*'  # расположение тестового набора данных
img_path_test_pn = 'img/test/PNEUMONIA_SMALL_half/*'  # расположение тестового набора данных
img_test_normal_np = []  # массив для хранения изображений здоровых пациентов
img_test_pn_np = []  # массив для хранения изображений
prediction_normal = []  # результаты распознавания изображений здоровых пациентов
prediction_pn = []  # результаты распознавания изображений больных пациентов

for i in glob.glob(img_path_test_nrm):  # считываем изображения здоровых пациентов
    result = training.read_img(i)
    img_test_normal_np.append(result)

for i in glob.glob(img_path_test_pn):  # считываем изображения больных пациентов
    result = training.read_img(i)
    img_test_pn_np.append(result)

for i in img_test_normal_np):  # распознаем изображения здоровых пациентов
    result = training.neural_network(i, weight_test)
    prediction_normal.append(result)

for i in img_test_pn_np:  # распознаем изображения больных пациентов
    result = training.neural_network(i, weight_test)
    prediction_pn.append(result)

# сохраняем результаты распознавания
np.savetxt('result_recognition_normal.csv', prediction_normal, delimiter=',', fmt='%.5f')
np.savetxt('result_recognition_pn.csv', prediction_pn, delimiter=',', fmt='%.5f')