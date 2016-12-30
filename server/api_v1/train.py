import os
import time
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from pprint import pprint
from keras.preprocessing import sequence
from keras.models import Sequential
from keras.optimizers import SGD
from keras.layers import Embedding, Dense, LSTM, TimeDistributed, Dropout, Activation
from keras.utils.visualize_util import plot
import json
from flask import request, jsonify, g
import flask_restful as restful
from server import mongo
from datetime import timedelta, datetime
import numpy as np
import csv
from sklearn import linear_model

class Train(restful.Resource):
    def get(self):
        size = 50000
        rolling_window_size = 5
        Y = []
        X = []
        data = list(mongo.db.neural_data.find({}).sort('timestamp', -1).limit(50000))
        for i, ex in enumerate(data[10:]):
            # this_minute_pseudo = data[i]["pseudo_log_return"] * 1000
            this_minute_price_growth = data[i]["price_growth"]
            arr = create_rolling_window(data, i, rolling_window_size)
            Y.append(this_minute_price_growth)
            X.append(arr)



        X_train = np.array(X[:int(size * 0.85)])
        X_test = np.array((X[int(size * 0.85):]))

        Y_train = np.array(Y[:int(size * 0.85)])
        Y_test = np.array(Y[int(size * 0.85):])
        # from aetros.KerasIntegration import KerasIntegration
        # API_KEY="7f7a67efc5233edc4ab067a7d518c94b" python main.py for aetros
        epochs = 1
        batch_size = 1
        ts = 1
        results = []
        num_features = rolling_window_size * 4
        # train data
        # 9207 - === 9088 / 32 = 71
        # X_train = np.genfromtxt('./data/yahoo.spy.train.csv', delimiter=', ', skip_header=1005, usecols=range(0,num_features))
        # Y_train = np.genfromtxt('./data/yahoo.spy.train.csv', delimiter=', ', skip_header=1005, usecols=(num_features))

        # test data
        # X_test = np.genfromtxt('./data/yahoo.spy.test.csv', delimiter=', ', skip_header=1,  usecols=range(0,num_features))
        # Y_test = np.genfromtxt('./data/yahoo.spy.test.csv', delimiter=', ', skip_header=1, usecols=(num_features))
        # validation_data
        # X_validate = np.genfromtxt('./data/yahoo.spy.validation.csv', delimiter=', ', skip_header=1, skip_footer=1, usecols=range(0, num_features))
        # Y_validate = np.genfromtxt('./data/yahoo.spy.validation.csv', delimiter=', ', skip_header=1, usecols=(num_features))

        # X's
        print(X_train.shape)
        X_train = np.reshape(X_train, (X_train.shape[0], 1,  X_train.shape[1]))
        X_test = np.reshape(X_test, (X_test.shape[0], 1, X_test.shape[1]))
        # X_validate = np.reshape(X_validate, (X_validate.shape[0], 1, X_validate.shape[1]))
        # Y's
        Y_train = np.reshape(Y_train, (Y_train.shape[0], 1))
        Y_test = np.reshape(Y_test, (Y_test.shape[0], 1))
        # Y_validate = np.reshape(Y_validate, (Y_validate.shape[0], 1))

        print("X SHAPE:", X_train.shape, X_train.ndim)
        print("Y SHAPE:", Y_train.shape, Y_train.ndim)
        print("X SAMPLE:", X_train[0][0])
        print("Y SAMPLE:", Y_train[0][0])


        model = Sequential()
        # model.add(Dense(6,  activation='tanh', ))
        # print(model.input_shape, model.output_shape)
        model.add(LSTM(20, batch_input_shape=(batch_size, ts, num_features),
                       activation="tanh",
                       return_sequences=True,
                       stateful=True))
        print(model.output_shape)

        model.add(LSTM(20,
                        activation="tanh",
                        return_sequences=False,
                        stateful=True))

        print(model.output_shape)
        # model.add(Dense(32, input_shape=(num_features,), activation="tanh"))
        # model.add(Dense(32, activation="tanh"))
        model.add(Dense(16, activation="tanh"))
        print(model.output_shape)
        model.add(Dense(12, activation="tanh"))
        print(model.output_shape)
        model.add(Dense(8, activation="tanh"))

        print(model.output_shape)
        # model.add(Dense(2, activation="linear"))
        # print(model.output_shape)
        # model.add(Dense(6, activation="tanh"))
        # print(model.output_shape)
        # model.add(Dense(3, activation="linear"))
        # print(model.output_shape)
        model.add(Dense(1, activation="linear"))
        print(model.output_shape)
        # model.add(Dense(20, activation="tanh"))
        # print(model.output_shape)
        # model.add(Dense(10, activation="linear"))
        # print(model.output_shape)
        # model.add(Dense(5, activation="linear"))
        # print(model.output_shape)
        # model.add(Dense(1, activation="linear"))
        # print(model.output_shape)
        # loss function, optimizer, metric
        model.compile(loss='mse',
                      metrics=["accuracy", 'mae'],
                      optimizer="adamax")
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
        scores = model.evaluate(x=X_test, y=Y_test, batch_size=batch_size)
        predicted_output = model.predict(X_test, batch_size=batch_size, verbose=1)
        # save model in json
        # results
        predicted_output = predicted_output.tolist()
        print('Test score:', scores[0])
        print('Test accuracy:', scores[1])
        print("HISTORY:", history)
        Y_test = Y_test.tolist()
        try:
            for x in predicted_output:
                if x[0] >= 0.5:
                    x[0] = 1
                elif x[0] < 0.5:
                    x[0] = 0

            temp_results = open('./data/temp_results.txt', 'r+')
            temp_results.write("Real Predicted\n")

            failed = 0
            for i, x in enumerate(Y_test):
                z = int(x[0])
                y = int(predicted_output[i][0])
                print(z, y)
                temp_results.write("{}, {}\n".format(z, y))
                if y != z:
                    failed += 1
            temp_results.close()
            print('FAILED: {}'.format(failed))
            print('CORRECT: {}'.format(int(len(Y_test) - failed)))
            print("ACCURACY: {}".format(failed / len(Y_test)))
        except Exception as e:
            print(e)
        print('PLOTTING RESULTS:')
        plt.plot(Y_test[:100], 'blue')
        plt.plot(predicted_output[:100], 'red')
        plt.show()
        plot(model, to_file='model.png')

        model.save_weights("model.h5")
        print("Saved model to disk")
        json_string = model.to_json()
        json_file = open('model.json', 'r+')
        json_file.write(json_string)


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

def create_rolling_window(data, i, size):
    arr = []
    for inner in range(1, size + 1):
        # minute_score = data[i - inner]["z_score"]
        std = data[i - inner]["std"]
        price_growth = data[i - inner]["price_growth"]
        pseudo_log_return = data[i - inner]["pseudo_log_return"] * 1000
        log_return = data[i - inner]["log_return"] * 1000
        arr.append(log_return)
        arr.append(pseudo_log_return)
        arr.append(std)
        arr.append(price_growth)
    return arr
