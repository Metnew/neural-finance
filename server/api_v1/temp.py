import flask_restful as restful
from server import mongo
from datetime import datetime
import numpy as np
import time
import math
from pprint import pprint
from sklearn.preprocessing import scale, MinMaxScaler, normalize

class Neural_Data(restful.Resource):
    # divide our finance collections in train, test datasets
    # Train = 70 %
    # Test = 30%

    def get(self):

        normalize_data_for_NN('GSPC')
        normalize_data_for_NN('DJI')
        return "DATA STORED!"


def z_score(value, obj):
    return (value - obj["mean"]) / obj["std"]
#
# def toFixed(num):
#     return  "{:10.2f}".format(num)

def min_max(X):
    min_max_scaler = MinMaxScaler()
    X = scale(X, axis=0, with_mean=True, with_std=True, copy=True )
    X = min_max_scaler.fit_transform(X)
    return X

def sk_normalize(X):
    return normalize(X) #, norm='l2'

def sk_scale(X):
    return scale(X, axis=0, with_mean=True, with_std=True, copy=True )


def normalize_data_for_NN(index_name):
    unix_now = time.time()  # time right now
    print('NORMALIZING DATA...')
    # Get ALL finance data for this index_name
    make_dataset(index_name)
    print("DATA NORMALIZED in {} seconds".format(time.time() - unix_now))


def make_dataset(index_name):

    finance_data = mongo.db["finance_data_" + index_name]
    collection = mongo.db["neural_data_" + index_name]
    collection.remove()  # remove past data from collection
    dataset_size = finance_data.find({}).count()
    last_data_timestamp = 1
    # np.array([])
    std_arr = np.array([], ndmin=1)  # std of prices on every minute
    time_sec = np.array([], ndmin=1)  # prices :ss on every minute
    time_hour = np.array([], ndmin=1)  # prices :HH on every minute
    time_minute = np.array([], ndmin=1)  # prices :MM on every minute
    time_unix = np.array([], ndmin=1)  # timestamp of every minute
    average_price_arr = np.array([], ndmin=1)  # average_price of every minute
    close_price_arr = np.array([], ndmin=1)  # average_price of every minute
    # last_data_timestamp = 1  # for DB queries, look at "data" variable
    last_average = 1  # last average price defined outside the loop scope
    last_close = 1  # last close price defined outside the loop scope
    log_return_arr = np.array([], ndmin=1)  # logarithmic difference of close prices of every consecutive minutes
    # same as log_return, but average price instead of close_price
    pseudo_log_return_arr = np.array([], ndmin=1)
    # price_growth_arr = []  # more about "price_growth" below
    price_growth_percent_arr = np.array([], ndmin=1)  # more about "price_growth_percent" below

    count = dataset_size
    while count > 0:
        # we can't get all data in one request, because we can have more than 400k
        # of records in every collection, so => limit(1000)
        limit = 100000
        if count < limit:
            limit = count
        data = list(finance_data.find({"timestamp": {"$gte": last_data_timestamp}}).sort(
            "timestamp", 1).limit(limit))
        count -= limit
        print(count)
        last_data_timestamp = data[limit - 1]["timestamp"] + 1
        for x in data:
            # x is a minute of trading
            open_price = x["open_price"]
            close_price = x["close_price"]
            low_price = x["low_price"]
            high_price = x["high_price"]
            start_date = x["timestamp"]

            prices = np.array([open_price, low_price, close_price, high_price])
            # average price of minute
            average = np.average(prices)

            local_time = datetime.fromtimestamp(start_date)
            unix_timestamp = local_time.timestamp()
            sec = int(local_time.strftime("%S"))
            minute = int(local_time.strftime("%M"))
            # we can count only active trading hours
            hour = int(local_time.strftime("%H")) - 13

            # binary classification
            # if close price bigger than close price of last minute,
            # then minute label is 1, else - 0
            # if close_price - last_close > 0:
            #     price_growth = 1
            # elif close_price - last_close < 0:
            #     price_growth = 0

            price_growth_percent = (close_price - last_close) / last_close
            # price growth compared to prev minute
            # price_growth_percent = (close_price - last_close) / last_close
            # price growth compared to prev minute
            pseudo_log_return = np.log(average / last_average)
            log_return = np.log(close_price / last_close)

            last_average = average
            last_close = close_price

            pseudo_log_return_arr = concat_timeline_value(pseudo_log_return_arr, pseudo_log_return)
            price_growth_percent_arr = concat_timeline_value(price_growth_percent_arr, price_growth_percent)
            average_price_arr = concat_timeline_value(average_price_arr, average)
            close_price_arr = concat_timeline_value(close_price_arr, close_price)
            log_return_arr = concat_timeline_value(log_return_arr, log_return)
            time_minute = concat_timeline_value(time_minute, minute)
            time_hour = concat_timeline_value(time_hour, hour)
            time_unix = concat_timeline_value(time_unix, unix_timestamp)
            std_arr = concat_timeline_value(std_arr, np.std(prices))
            # concat_timeline_value(time_unix, unix_timestamp)


            # pseudo_log_return_arr = np.stack([pseudo_log_return_arr, pseudo_log_return])
            # price_growth_arr.append(price_growth)
            # price_growth_percent_arr = (price_growth_percent)
            # average_price_arr = (average)
            # close_price_arr = (close_price)
            # log_return_arr = (log_return)
            # time_minute = (minute)
            # time_hour = (hour)
            # time_sec = (sec)
            # time_unix = unix_timestamp
            # std_arr = np.std(prices)
            # index_for_numpy_iter += 1

    # I'm not sure that this function can remove outliers
    # neural_data = reject_outliers(neural_data) # remove outliers
    cbs = [sk_scale, sk_normalize, min_max]
    # price_growth_percent_arr = compose(price_growth_percent)
    average_price_arr = compose(average, cbs)
    close_price_arr = compose(close_price, cbs)
    log_return_arr = compose(log_return, cbs)
    std_arr = compose(std_arr, cbs)

    index_in_norm_arr = 0
    # X = np.array([average_price_arr, close_price_arr, log_return_arr, pseudo_log_return_arr, std_arr, price_growth_percent_arr])
    # X = np.rot90(X)
    X = np.vstack((average_price_arr, close_price_arr, log_return_arr, pseudo_log_return_arr, std_arr, price_growth_percent_arr))
    # X = sk_scale(X)
    # X = sk_normalize(X)
    # X = min_max(X)

    store_mongo_arr = []
    for i in range(0, dataset_size):
        if i == 0: # price_growth_percent is invalid for 0 element
            continue

        minute = {
            "std": X[i][4],
            'sec': time_sec[i],
            'minute': time_minute[i],
            'hour': time_hour[i],
            'timestamp': time_unix[i],
            'pseudo_log_return': X[i][3],
            'log_return': X[i][2],
            'price_growth_percent': X[i][5],
            'close_price': X[i][1],
            'average_prices': X[i][0]
        }

        store_mongo_arr.append(minute)  # insert minute in arr
        if len(store_mongo_arr) == 5000:
            collection.insert_many(store_mongo_arr)
            store_mongo_arr = []

    return None

def compose(x, arr):
    for func in arr:
        x = func(x)
    return x

def concat_timeline_value(arr, value):
    return np.stack([arr, value])
