import matplotlib.pyplot as plt
import numpy as np
name = 'results-2017-05-15 22:38:35'
Y = np.genfromtxt('{}.txt'.format(name), delimiter=',', usecols=(0))
predict = np.genfromtxt('{}.txt'.format(name), delimiter=',', usecols=(1))

# plt.plotfile('results-2017-05-14 01:07:23.txt', ('y', 'predicted'), delimiter=',', subplots=False)
plt.plot(Y[400:500], c='blue')
plt.plot(predict[400:500], c="red")
plt.show()
