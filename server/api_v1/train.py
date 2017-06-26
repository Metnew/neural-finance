import os
import time
import numpy as np
import matplotlib.pyplot as plt
from time import gmtime, strftime
from pprint import pprint
from keras.models import Sequential
from keras.layers import Dense, LSTM, Dropout, Activation
from keras.utils import plot_model
import flask_restful as restful
from sklearn.preprocessing import scale, MinMaxScaler, normalize, robust_scale, maxabs_scale, minmax_scale
from server import mongo
import numpy as np
import math
from sklearn.metrics import r2_score


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
        count = 100000
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
        rolling_window_size = 1
        print("Start data processing!")
        # every rolling window - is input vector for network
        for i, ex in enumerate(data[10:]):
            # std = data[i]["std"]
            # pseudo = data[i]["pseudo_log_return"]
            price_growth_percent_normalized = data[i]["pseudo_log_return"]

            arr = create_rolling_window(data, i, rolling_window_size)
            # this_minute_y = [pseudo, std, price_growth_percent_normalized]
            this_minute_y = [price_growth_percent_normalized]
            Y.append(this_minute_y)
            X.append(arr)

        print("Numpy arrays are ready.")
        length = math.ceil(len(X) * 0.8)

        X_train = np.array(X[:length])
        X_test = np.array(X[length:])
        Y_train = np.array(Y[:length])
        Y_test = np.array(Y[length:])

        epochs = 1
        batch_size = 1
        results = []
        ts = 1
        features_in_vector = 2 #12
        num_features = 2
        # reshape for LSTM
        X_train = np.reshape(X_train, (X_train.shape[0], 1, 2))
        X_test = np.reshape(X_test, (X_test.shape[0], 1,  2))


        print("X SHAPE:", X_train.shape, X_train.ndim)
        print("Y SHAPE:", Y_train.shape, Y_train.ndim)
        print("X SAMPLE:", X_train[0])
        print("Y SAMPLE:", Y_train[0])

        model = Sequential()
        # model.add(Dense(6, input_shape=(num_features,), activation="tanh"))

        model.add(LSTM(3, batch_input_shape=(batch_size, ts, num_features),
                       activation="tanh",
                       return_sequences=True,
                       stateful=True))
        # model.add(LSTM(3,
        #                activation="tanh",
        #                return_sequences=True,
        #                stateful=True))
        model.add(LSTM(3,
                       activation="tanh",
                       return_sequences=False,
                       stateful=True))
        # model.add(Dense(16, activation="tanh"))
        # model.add(Dense(8, activation="tanh"))
        # model.add(Dense(5, activation="tanh"))
        model.add(Dense(1, activation="linear"))
        # loss function, optimizer, metric
        model.compile(loss='mse', metrics=["mae", "mape"], optimizer="adam")
        # fit the model
        print("TRAINING...")
        history = model.fit(X_train, Y_train,
                            batch_size=batch_size,
                            epochs=epochs,
                            verbose=1)
        # history = history.history
        # Evaluate accuracy
        # scores = model.evaluate(x=X_test, y=Y_test, batch_size=batch_size)
        # results
        # print('Test score:', scores[0])
        # print('Test accuracy:', scores[1])
        # print("HISTORY:", history)

        tme = strftime("%Y-%m-%d %H:%M:%S", gmtime())


        # Y_test = Y_test.tolist()
        predicted_output = model.predict(X_test, batch_size=batch_size, verbose=1)
        print('R2 SCORE:', r2_score(Y_test, predicted_output))
        ##########################################################
        # Y_test_zero_to_one = sk_min_max(Y_test)
        # predicted_output_zero_to_one = sk_min_max(predicted_output)
        # print('R2 SCORE:', r2_score(Y_test_zero_to_one, predicted_output_zero_to_one))
        ##########################################################
        All = np.hstack([Y_test, predicted_output])
        np.savetxt('results-{}.txt'.format(tme), All, delimiter=',')
        print('PLOTTING RESULTS:')

        plt.plot(Y_test[:100], 'blue')
        plt.plot(predicted_output[:100], 'red')
        plt.savefig("results-{}.png".format(tme))
        plt.ion()
        plt.show()

def create_rolling_window(data, i, size):
    arr = []
    for inner in range(1, size + 1):
        step = []
        std = data[i - inner]["std"]
        # close_price = data[i - inner]["close_price"]
        # average_prices = data[i - inner]["average_prices"]
        # price_growth_percent = data[i - inner]["price_growth_percent_normalized"]
        pseudo_log_return = data[i - inner]["pseudo_log_return"]
        # log_return = data[i - inner]["log_return"]

        # arr.append(log_return)
        arr.append(pseudo_log_return)
        arr.append(std)

        # arr.append(close_price)
        # arr.append(price_growth_percent)
        # arr.append(step)
        # arr.append(price_growth_percent)
        # arr.append(price_growth)

    return arr

def toFixed(num):
    return  "{:10.2f}".format(num)

def sk_min_max(X):
    min_max_scaler = MinMaxScaler()
    # X = scale(X, axis=0, with_mean=True, with_std=True, copy=True)
    return min_max_scaler.fit_transform(X)
