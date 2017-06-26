import matplotlib.pyplot as plt
import numpy as np
from sklearn.preprocessing import scale, MinMaxScaler, normalize, robust_scale, maxabs_scale, minmax_scale
from sklearn.metrics import r2_score, mean_squared_error as mse, mean_absolute_error as mae

name = './records/results-2017-05-15 10:15:23'
Y = np.genfromtxt('{}.txt'.format(name), delimiter=',', usecols=(0))
predict = np.genfromtxt('{}.txt'.format(name), delimiter=',', usecols=(1))
# Y = scale(Y, with_std=True)
# predict = scale(predict, with_std=True)

# Y = minmax_scale(Y, feature_range=(-1, 1))
# predict = minmax_scale(predict, feature_range=(-1, 1))

Y = Y[0:100]
predict = predict[0:100]
Y_prev = 0
predict_prev = 0
point_to_griphindor = 0
point_to_slizerin = 0

for index, val in enumerate(Y):
    if Y_prev >= Y[index] and predict_prev >= predict[index]:
        point_to_griphindor+=1
    elif Y_prev <= Y[index] and predict_prev <= predict[index]:
        point_to_griphindor+=1
    else:
        point_to_slizerin+=1

    Y_prev = Y[index]
    predict_prev = predict[index]
print(point_to_slizerin, point_to_griphindor)

print('R2 SCORE:', r2_score(Y, predict))
print('MSE SCORE:', mse(Y, predict))
print('MAE SCORE:', mae(Y, predict))
# print('MAPE SCORE:', r2_score(Y, predict))

plt.plot(Y, c='blue')
plt.plot(predict, c="red")
plt.show()
