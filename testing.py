
import training
import numpy as np
import glob


'''
Тестирование на обучающем наборе данных.
a = 5
# тестируем нейросеть
print('сначала нормальные')
for i in img_train_normal_np:
    prediction = neural_network(i, weights)
    print('prediction = ', prediction)
    a -= 1
    if a == -1:
        break

a = 5
print('теперь ненормальные')
for i in img_train_pn_np:
    prediction = neural_network(i, weights)
    print('prediction = ', prediction)
    a -= 1
    if a == -1:
        break
'''


# Тестирование на тестовом наборе данных.


weight_test = np.loadtxt('text.csv', delimiter=',')

img_path_test_begin = 'img/test/'
img_path_test_nrm_end = 'NORMAL_SMALL_half/*'
img_path_test_pn_end = 'PNEUMONIA_SMALL_half/*'

img_test_normal_name = []
img_test_normal_np = []
for i in glob.glob(img_path_test_begin + img_path_test_nrm_end):
    img_test_normal_name.append(i)
    img_test_normal_np.append(training.read_img(i))

img_test_pn_name = []
img_test_pn_np = []
for i in glob.glob(img_path_test_begin + img_path_test_pn_end):
    img_test_pn_name.append(i)
    img_test_pn_np.append(training.read_img(i))

prediction_normal = []
for i in img_test_normal_np:
    result = training.neural_network(i, weight_test)
    prediction_normal.append(result)

prediction_pn = []
for i in img_test_pn_np:
    result = training.neural_network(i, weight_test)
    prediction_pn.append(result)

np.savetxt('normal2.csv', prediction_normal, delimiter=',', fmt='%.5f')
np.savetxt('pn2.csv', prediction_pn, delimiter=',', fmt='%.5f')





