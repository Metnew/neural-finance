import os
import time
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from pprint import pprint
from keras.preprocessing import sequence
from keras.models import Sequential, model_from_json
from keras.optimizers import SGD
from keras.layers import Embedding, Dense, LSTM, TimeDistributed, Dropout, Activation
from keras.utils.visualize_util import plot
# from aetros.KerasIntegration import KerasIntegration
# API_KEY="7f7a67efc5233edc4ab067a7d518c94b" python main.py for aetros
epochs = 1
batch_size = 1
ts = 1
results = []
num_features = 10
# train data
# 9207 - === 9088 / 32 = 71
X_train = np.genfromtxt('./data/yahoo.spy.train.csv', delimiter=', ', skip_header=1005, usecols=range(0, num_features))
Y_train = np.genfromtxt('./data/yahoo.spy.train.csv', delimiter=', ', skip_header=1005, usecols=(num_features))
# test data
# X_test = np.genfromtxt('./data/yahoo.spy.test.csv', delimiter=', ', skip_header=1,  usecols=range(0,num_features))
# Y_test = np.genfromtxt('./data/yahoo.spy.test.csv', delimiter=', ', skip_header=1, usecols=(num_features))
# validation_data
X_validate = np.genfromtxt('./data/yahoo.spy.validation.csv', delimiter=', ', skip_header=1, skip_footer=1, usecols=range(0, num_features))
Y_validate = np.genfromtxt('./data/yahoo.spy.validation.csv', delimiter=', ', skip_header=1, usecols=(num_features))

# X's
# print(X_test[0])
X_train = np.reshape(X_train, (X_train.shape[0], 1,  X_train.shape[1]))
# X_test = np.reshape(X_test, (X_test.shape[0], 1, X_test.shape[1]))
X_validate = np.reshape(X_validate, (X_validate.shape[0], 1, X_validate.shape[1]))
# Y's
Y_train = np.reshape(Y_train, (Y_train.shape[0], 1))
# Y_test = np.reshape(Y_test, (Y_test.shape[0], 1))
Y_validate = np.reshape(Y_validate, (Y_validate.shape[0], 1))

print("X SHAPE:", X_train.shape, X_train.ndim)
print("Y SHAPE:", Y_train.shape, Y_train.ndim)
print("X SAMPLE:", X_train[0][0])
print("Y SAMPLE:", Y_train[0][0])


model = Sequential()
# model.add(Dense(6,  activation='tanh', ))
# print(model.input_shape, model.output_shape)
model.add(LSTM(11, batch_input_shape=(batch_size, ts, num_features),
               activation="tanh",
               return_sequences=True,
               stateful=True))
print(model.output_shape)

model.add(LSTM(11,
                activation="tanh",
                return_sequences=False,
                stateful=True))

print(model.output_shape)

model.add(Dense(8, activation="tanh"))
print(model.output_shape)
model.add(Dense(6, activation="tanh"))
print(model.output_shape)
model.add(Dense(4, activation="tanh"))
print(model.output_shape)
model.add(Dense(2, activation="tanh"))
print(model.output_shape)

model.add(Dense(1, activation="linear"))
print(model.output_shape)
# loss function, optimizer, metric
model.compile(loss='mse',
              metrics=["accuracy", 'mape', 'mae'],
              optimizer="adam")
# start time for measuring
start_time = time.time()
########AETROS WEB CLIENT#########
##################################
##################################
# from aetros.KerasIntegration import KerasIntegration
# KerasIntegration('Metnew/finbot', model, insights=True)
########AETROS WEB CLIENT#########
##################################
##################################
# fit the model
print("TRAINING")
history = model.fit(X_train, Y_train,
                    batch_size=batch_size,
                    nb_epoch=epochs,
                    verbose=1)
history = history.history
pprint(history)
# push to arr
# results.append(history)

# Evaluate accuracy
# score, acc = model.evaluate(X_test, Y_test, batch_size=batch_size)
predicted_output = model.predict(X_train, batch_size=batch_size, verbose=1)
# save model in json
# results
# print('\nTest score:', score)
# print('Test accuracy:', acc)
print("HISTORY:", history)
# temp_results = open('./data/temp_results.txt', 'r+')
# def logit(x):
#     return np.log(x / 1 - x)
# for x, y in np.nditer([Y_test, predicted_output]):
    # x = logit(x)
    # y = logit(y)
    # temp_results.write("{}, {}\n".format(x, y))
# temp_results.close()
print('PLOTTING RESULTS:')
plt.plot(Y_train, 'blue')
plt.plot(predicted_output, 'red')
plt.show()
plot(model, to_file='model.png')

# plt.plot(history.history['acc'])
# plt.plot(history.history['val_acc'])
# plt.title('model accuracy')
# plt.ylabel('accuracy')
# plt.xlabel('epoch')
# plt.legend(['train', 'test'], loc='upper left')
# plt.show()
# # summarize history for loss
# plt.plot(history.history['loss'])
# plt.plot(history.history['val_loss'])
# plt.title('model loss')
# plt.ylabel('loss')
# plt.xlabel('epoch')
# plt.legend(['train', 'test'], loc='upper left')
# plt.show()
