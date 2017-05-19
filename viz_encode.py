import matplotlib.pyplot as plt
import numpy as np
name = 'results-2017-05-15 02:44:52'
Y = np.genfromtxt('{}.txt'.format(name), delimiter=',', usecols=(1))
predict = np.genfromtxt('{}.txt'.format(name), delimiter=',', usecols=(4))

# plt.plotfile('results-2017-05-14 01:07:23.txt', ('y', 'predicted'), delimiter=',', subplots=False)
plt.plot(Y[:500], c='blue')
plt.plot(predict[:500], c="red")
plt.show()
