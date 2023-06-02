import training
import numpy as np
import glob


# загружаем весовые коэффициенты
weight_test = np.loadtxt('weights.csv', delimiter=',')
 
# расположение тестового набора данных
img_path_test_nrm = 'img/test/NORMAL_SMALL_half/*'
img_path_test_pn = 'img/test/PNEUMONIA_SMALL_half/*'

img_test_normal_np = []  # считываем изображения
for i in glob.glob(img_path_test_nrm):
    result = training.read_img(i)
    img_test_normal_np.append(result)

img_test_pn_np = []  # считываем изображения
for i in glob.glob(img_path_test_pn):
    result = training.read_img(i)
    img_test_pn_np.append(result)

prediction_normal = []  # результаты распознавания изображений здоровых
for i in range(len(img_test_normal_np)):
    result = training.neural_network(img_test_normal_np[i], weight_test)
    prediction_normal.append(result)

prediction_pn = []  # результаты распознавания изображений больных
for i in img_test_pn_np:
    result = training.neural_network(i, weight_test)
    prediction_pn.append(result)

# сохраняем результаты распознавания
np.savetxt('result_recognition_normal.csv', prediction_normal, delimiter=',', fmt='%.5f')
np.savetxt('result_recognition_pn.csv', prediction_pn, delimiter=',', fmt='%.5f')





