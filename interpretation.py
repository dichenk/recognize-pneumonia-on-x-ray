import numpy as np  # пакет для работы с массивами данных

# загружаем результаты распознавания
normal_test = np.loadtxt('result_recognition_normal.csv', delimiter=',')
pn_test = np.loadtxt('result_recognition_pn.csv', delimiter=',')


board = 0.9  # граница достоверности

normal_prob = 0  # вероятность правильного диагноза при существующей границе
pn_prob = 0  # вероятность правильного диагноза при существующей границе

for i in normal_test:  # для нормальных снимков
    if i[0] > board:
       normal_prob += 1

for i in pn_test:  # для снимков заболевших пациентов
    if i[1] > board:
       pn_prob += 1

normal_prob /= len(normal_test)
pn_prob /= len(pn_test)

print(f'Вероятность распознавания здорового ребенка при границе {board} равна {normal_prob}')
print(f'Вероятность распознавания больного ребенка при границе {board} равна {pn_prob}')