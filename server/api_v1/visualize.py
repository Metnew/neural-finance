import flask_restful as restful
from server import mongo
from datetime import datetime
import numpy as np
import time
import math
from pprint import pprint
from sklearn.preprocessing import scale, MinMaxScaler, normalize
import numpy as np
import matplotlib.pyplot as plt
# from scipy.stats.stats import pearsonr

class Visualize(restful.Resource):
    # divide our finance collections in train, test datasets
    # Train = 70 %
    # Test = 30%

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
        print("Start data processing!")
        # every rolling window - is input vector for network
        X = []
        log_return = []
        pseudo_log_return = []
        std = []
        percent = []
        unix = []
        average_prices = []
        close_price = []
        for v in data:
            unix.append(v["timestamp"])
            std.append(v["std"])
            percent.append(v["price_growth_percent"])
            pseudo_log_return.append(v["pseudo_log_return"])
            log_return.append(v["log_return"])
            average_prices.append(v["average_prices"])
            close_price.append(v["close_price"])


        log_return = np.array([log_return])
        pseudo_log_return = np.array([pseudo_log_return])
        percent = np.array([percent])
        std = np.array([std])
        average_prices = np.array([average_prices])
        close_price = np.array([close_price])

        log_std = np.corrcoef(log_return,std)
        percent_std = np.corrcoef(percent,std)
        pseudo_percent = np.corrcoef(pseudo_log_return,percent)
        log_percent = np.corrcoef(log_return,percent)
        close_percent = np.corrcoef(close_price, percent)
        pseudo_average = np.corrcoef(pseudo_log_return,average_prices)
        # pseudo_percent = numpy.corrcoef(pseudo_log_return,percent)
        # pseudo_percent = numpy.corrcoef(pseudo_log_return,percent)
        print(log_std)
        print(percent_std)
        print(pseudo_percent) # 0.8
        print(pseudo_average)
        print(close_percent) # 0.83


        print('PLOTTING RESULTS:')
        # plt.xlabel('time (s)')
        plt.figure(1)
        plt.subplot(221)
        plt.plot(std[:1000], 'blue')
        plt.subplot(222)
        plt.plot(percent[:1000], 'green')
        plt.subplot(223)
        plt.plot(pseudo_log_return[:1000], 'yellow')
        plt.subplot(224)
        plt.plot(log_return[:1000], 'red')

        # plt.figure(2)
        # plt.plot(pseudo_log_return[:1000], 'red')
        plt.savefig("test.png")
        plt.show()
