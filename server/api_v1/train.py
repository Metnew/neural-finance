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
import math


class Train(restful.Resource):

    def get(self):
        db = mongo.db
        index_name = 'GSPC'
        collection = db["neural_data_" + index_name]
        last_data_timestamp = 1

        # minutes
        X = []
        Y = []
        data = []
        # neural_train_data_DJI
        #
        # get data from mongo, mongo cursor to list
        dataset_size = collection.find({}).count()
        count = dataset_size
        while count > 0:
            # we can't get all data in one request, because we can have more than 400k
            # of records in every collection, so => limit(1000)
            limit = 50000
            if count < limit:
                limit = count
            data = data + list(collection.find({"timestamp": {"$gte": last_data_timestamp}}).sort(
                "timestamp", 1).limit(limit))
            count -= limit
            print(count, len(data))
            last_data_timestamp = data[limit - 1]["timestamp"] + 1


        # look at "create_rolling_window" function below
        # rolling window includes values of last ${rolling_window_size, e.g. 5}
        rolling_window_size = 5
        print("Start data processing!")
        # every rolling window - is input vector for network
        for i, ex in enumerate(data[10:]):
            this_minute_price_growth = data[i]["pseudo_log_return"]
            arr = create_rolling_window(data, i, rolling_window_size)
            this_minute_y = [this_minute_price_growth]
            Y.append(this_minute_y)
            X.append(arr)


        print("Numpy arrays are ready.")
        length = math.ceil(len(X) * 0.8)

        X_train = np.array(X[:length])
        X_test = np.array(X[length:])
        Y_train = np.array(Y[:length])
        Y_test = np.array(Y[length:])

        epochs = 5
        batch_size = 32
        results = []
        features_in_vector = 5 #12
        num_features = rolling_window_size * features_in_vector
        # reshape for LSTM
        # X_train = np.reshape(X_train, (X_train.shape[0], 1,  X_train.shape[1]))


        print("X SHAPE:", X_train.shape, X_train.ndim)
        print("Y SHAPE:", Y_train.shape, Y_train.ndim)
        print("X SAMPLE:", X_train[0])
        print("Y SAMPLE:", Y_train[0])

        model = Sequential()
        model.add(Dense(25, input_shape=(num_features,), activation="tanh"))
        # model.add(Dense(25, activation="tanh"))
        # model.add(LSTM(25, batch_input_shape=(batch_size, ts, num_features),
        #                activation="tanh",
        #                return_sequences=True,
        #                stateful=True))
        # model.add(LSTM(25,
        #                activation="tanh",
        #                return_sequences=False,
        #                stateful=True))
        model.add(Dense(16, activation="tanh"))
        model.add(Dense(9, activation="tanh"))
        # model.add(Dense(8, activation="tanh"))
        # model.add(Dense(4, activation="tanh"))
        model.add(Dense(1, activation="linear"))
        # loss function, optimizer, metric
        model.compile(loss='mape',
                      metrics=["accuracy", "mse", "mae"],
                      optimizer="sgd")
        # fit the model
        print("TRAINING...")
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

        # Y_test = Y_test.tolist()
        # predicted_output = model.predict(
            # X_train, batch_size=batch_size, verbose=1).tolist()

        # print('PLOTTING RESULTS:')
        # plt.plot(Y_train[:1000], 'blue')
        # plt.plot(predicted_output[:1000], 'red')
        # plt.show()
        # file_name = 'model-{}'.format(start_time)
        # plot(model, to_file=(file_name + '.png'))
        # model.save_weights("model" + ".h5")
        # print("Saved model to disk")
        # json_string = model.to_json()
        # json_file = open(file_name + '.json', 'w')
        # json_file.write(json_string)


def create_rolling_window(data, i, size):
    arr = []
    for inner in range(1, size + 1):
        std = data[i - inner]["std"]
        close_price = data[i - inner]["close_price"]
        average_prices = data[i - inner]["average_prices"]
        price_growth_percent = data[i - inner]["price_growth_percent"]
        pseudo_log_return = data[i - inner]["pseudo_log_return"]
        log_return = data[i - inner]["log_return"]

        arr.append(log_return)
        arr.append(pseudo_log_return)
        arr.append(std)

        # arr.append(average_prices)
        arr.append(close_price)
        arr.append(price_growth_percent)
        # arr.append(price_growth)

    return arr

def toFixed(num):
    return  "{:10.2f}".format(num)
