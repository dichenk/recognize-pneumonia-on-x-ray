
import numpy as np  # пакет для работы с массивами данных

normal_test = np.loadtxt('normal2.csv', delimiter=',')
pn_test = np.loadtxt('pn2.csv', delimiter=',')

# print(normal_test)

board = .85

normal_prob = 0
pn_prob = 0

for i in normal_test:
    if i[0] > board:
       normal_prob += 1 

for i in pn_test:
    if i[1] > board:
       pn_prob += 1

normal_prob /= len(normal_test)
pn_prob /= len(pn_test)

print(f'Вероятность определения здорового ребенка при границе {board} равна {normal_prob}')
print(f'Вероятность определения больного ребенка при границе {board} равна {pn_prob}')