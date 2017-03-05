import os
import time
import numpy as np
import matplotlib.pyplot as plt
from pprint import pprint
from keras.models import Sequential
from keras.layers import Dense, LSTM, Dropout, Activation
from keras.utils.visualize_util import plot
import flask_restful as restful
from server import mongo
import numpy as np


class Train(restful.Resource):

    def get(self):
        # start time for measuring
        start_time = time.time()
        size = 40000
        rolling_window_size = 10
        # look at "create_rolling_window" function below
        # rolling window includes values of last ${rolling_window_size, e.g. 5}
        # minutes
        Y = []
        X = []
        # get data from mongo, mongo cursor to list
        data = list(mongo.db.neural_data.find({}).sort('timestamp', -1).limit(size))
        # every rolling window - is input vector for network
        for i, ex in enumerate(data[10:]):
            this_minute_price_growth = data[i]["price_growth"]
            arr = create_rolling_window(data, i, rolling_window_size)
            this_minute_y = []
            if this_minute_price_growth == 0:
                this_minute_y = [0, 1]
            else:
                this_minute_y = [1, 0]
            Y.append(this_minute_y)
            X.append(arr)

        X_train = np.array(X[:int(size * 0.85)])  # 85% of data
        X_test = np.array((X[int(size * 0.85):]))  # 85% of data

        Y_train = np.array(Y[:int(size * 0.85)])  # 15% of data
        Y_test = np.array(Y[int(size * 0.85):])  # 15% of data
        epochs = 1
        batch_size = 1  # incremental learning
        ts = 1  # timestep (required for LSTM)
        results = []
        num_features = rolling_window_size * 4
        # X's
        # reshape for LSTM
        X_train = np.reshape(X_train, (X_train.shape[0], 1,  X_train.shape[1]))
        X_test = np.reshape(X_test, (X_test.shape[0], 1, X_test.shape[1]))
        # Y's
        # reshape
        Y_train = np.reshape(Y_train, (Y_train.shape[0], 2))
        Y_test = np.reshape(Y_test, (Y_test.shape[0], 2))

        print("X SHAPE:", X_train.shape, X_train.ndim)
        print("Y SHAPE:", Y_train.shape, Y_train.ndim)
        print("X SAMPLE:", X_train[0][0])
        print("Y SAMPLE:", Y_train[0])

        model = Sequential()
        model.add(LSTM(20, batch_input_shape=(batch_size, ts, num_features),
                       activation="tanh",
                       return_sequences=True,
                       stateful=True))
        model.add(LSTM(20,
                       activation="tanh",
                       return_sequences=False,
                       stateful=True))
        model.add(Dense(16, activation="tanh"))
        model.add(Dense(12, activation="tanh"))
        model.add(Dense(8, activation="tanh"))
        model.add(Dense(4, activation="tanh"))
        model.add(Dense(2, activation="softmax"))
        # loss function, optimizer, metric
        model.compile(loss='mse',
                      metrics=["accuracy", 'mae', 'mape'],
                      optimizer="nadam")
        # fit the model
        print("TRAINING")
        history = model.fit(X_train, Y_train,
                            batch_size=batch_size,
                            nb_epoch=epochs,
                            verbose=1)
        history = history.history
        # Evaluate accuracy
        scores = model.evaluate(x=X_test, y=Y_test, batch_size=batch_size)
        # results
        print('Test score:', scores[0])
        print('Test accuracy:', scores[1])
        print("HISTORY:", history)

        Y_test = Y_test.tolist()
        predicted_output = model.predict(
            X_test, batch_size=batch_size, verbose=1).tolist()
        try:
            # for x in predicted_output:
            #     if x[0] >= 0.5:
            #         x[0] = 1
            #     elif x[0] < 0.5:
            #         x[0] = 0

            temp_results = open('./data/temp_results.txt', 'r+')
            temp_results.write("Real Real Predicted Predicted\n")

            # failed = 0
            for i, x in enumerate(Y_test):
                pred = predicted_output[i]
                real = Y_test[i]
                real_and_pred = real + pred
                str_temp = ''.join(toFixed(e) for e in real_and_pred).strip()
                temp_results.write(str_temp + '\n')
                # if pred != real:
                #     failed += 1
            temp_results.close()
            # print('FAILED: {}'.format(failed))
            # print('CORRECT: {}'.format(int(len(Y_test) - failed)))
            # print("ACCURACY: {}".format((len(Y_test) - failed) / len(Y_test)))
        except Exception as e:
            print(e)
        print('PLOTTING RESULTS:')
        plt.plot(Y_test[:1000], 'blue')
        plt.plot(predicted_output[:1000], 'red')
        plt.show()
        file_name = 'model-{}'.format(start_time)
        plot(model, to_file=(file_name + '.png'))
        model.save_weights("model" + ".h5")
        print("Saved model to disk")
        json_string = model.to_json()
        json_file = open(file_name + '.json', 'w')
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

def toFixed(num):
    return  "{:10.2f}".format(num)
