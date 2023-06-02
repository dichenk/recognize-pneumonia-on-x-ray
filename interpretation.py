# import testing
import numpy as np  # пакет для работы с массивами данных

normal_test = np.loadtxt('result_normal_5.csv', delimiter=',')
pn_test = np.loadtxt('result_pn_5.csv', delimiter=',')

# names_normal = np.loadtxt('normal_name_5.csv', delimiter=',')
# names_pn = np.loadtxt('pn_name_5.csv', delimiter=',')

board = 0.7

normal_prob = 0
pn_prob = 0

for i in normal_test:
    if i[0] > board:
       normal_prob += 1
       print('раз')

for i in pn_test:
    if i[1] > board:
       pn_prob += 1

normal_prob /= len(normal_test)
pn_prob /= len(pn_test)

print(f'Вероятность определения здорового ребенка при границе {board} равна {normal_prob}')
print(f'Вероятность определения больного ребенка при границе {board} равна {pn_prob}')