
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


weight_test = np.loadtxt('text5.csv', delimiter=',')
 
img_path_test_begin = 'img/test/'
img_path_test_nrm_end = 'NORMAL_SMALL_half/*'
img_path_test_pn_end = 'PNEUMONIA_SMALL_half/*'

img_test_normal_np = []
name_normal = []
for i in glob.glob(img_path_test_begin + img_path_test_nrm_end):
    result = training.read_img(i)
    img_test_normal_np.append(result)
    name_normal.append(i)

img_test_pn_np = []
for i in glob.glob(img_path_test_begin + img_path_test_pn_end):
    result = training.read_img(i)
    img_test_pn_np.append(result)
    


prediction_normal = []
for i in range(len(img_test_normal_np)):
    result = training.neural_network(img_test_normal_np[i], weight_test)
    prediction_normal.append(result)
    if result[0] < .8: print(name_normal[i])


prediction_pn = []
for i in img_test_pn_np:
    result = training.neural_network(i, weight_test)
    prediction_pn.append(result)

np.savetxt('result_normal_5.csv', prediction_normal, delimiter=',', fmt='%.5f')
np.savetxt('result_pn_5.csv', prediction_pn, delimiter=',', fmt='%.5f')





