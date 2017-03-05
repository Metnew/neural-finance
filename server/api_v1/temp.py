import flask_restful as restful
from server import mongo
from datetime import datetime
import numpy as np
import time
import math

class Neural_Data(restful.Resource):
    # divide our finance collections in train, test datasets
    # Train = 70 %
    # Test = 30%

    def get(self):


        normalize_data_for_NN('GSPC')
        normalize_data_for_NN('DJI')
        return "DATA STORED!"


def z_score(value, mean, std):
    return (value - mean) / std

def normalize_data_for_NN(index_name):
    finance_data = mongo.db["finance_data_" + index_name]  # finance_data collection
    # neural_data TRAIN collection
    neural_train_data = mongo.db["neural_train_data_" + index_name]
    # neural_data TEST collection
    neural_test_data = mongo.db["neural_test_data_" + index_name]
    neural_train_data.remove() # clear train collection
    neural_test_data.remove()  # clear test collection

    neural_data = []  # array
    std = []  # std of prices on every minute
    time_sec = []  # prices :ss on every minute
    time_hour = []  # prices :HH on every minute
    time_minute = []  # prices :MM on every minute
    time_unix = []  # timestamp of every minute
    average_price_arr = []  # average_price of every minute

    last_data_timestamp = 1  # for DB queries, look at "data" variable
    last_average = 1  # last average price defined outside the loop scope
    last_close = 1  # last close price defined outside the loop scope
    # logarithmic difference of average prices of every consecutive
    # minutes
    log_return_arr = []  # logarithmic difference of close prices of every consecutive minutes
    pseudo_log_return_arr = [] # same as log_return, but average price instead of close_price
    price_growth_arr = []  # more about price growth below

    unix_now = time.time()  # time right now
    print('NORMALIZING DATA...')
    # Get ALL finance data for this index_name
    # we can't get all data in one request, cause we can have more than 400k
    # of records in every collection
    dataset_size = finance_data.find().count()
    train_size = math.floor(dataset_size * 0.7)
    test_size = dataset_size - train_size
    while dataset_size > 0:
        data = finance_data.find({"gte": {"timestamp": last_data_timestamp}}).sort(
            "timestamp", -1).limit(1000)
        dataset_size -= 1000
        last_data_timestamp = data[0]["timestamp"]
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
            sec = int(local_time.strftime("%S"))
            minute = int(local_time.strftime("%M"))
            hour = int(local_time.strftime("%H")) - 13

            # binary classification
            # if close price bigger than close price of last minute,
            # then minute label is 1, else - 0
            if close_price - last_close > 0:
                price_growth = 1
            elif close_price - last_close < 0:
                price_growth = 0

            pseudo_log_return = np.log(average / last_average)
            log_return = np.log(close_price / last_close)

            last_average = average
            last_close = close_price

            pseudo_log_return_arr.append(pseudo_log_return)
            price_growth_arr.append(price_growth)
            average_price_arr.append(average)
            log_return_arr.append(log_return)
            time_minute.append(minute)
            time_hour.append(hour)
            time_sec.append(sec)
            std.append(np.std(prices))

    # I'm not sure that this function can remove outliers
    # neural_data = reject_outliers(neural_data) # remove outliers


    # according to A.Karparthy course we have to use statistics from train set
    # to normalize test and validation set too
    # find mean price of train set
    neural_train_data_price_mean = np.average(neural_train_data)
    # find standart deviation of train set
    neural_train_data_std = np.std(neural_data)

    index_in_norm_arr = 0
    store_mongo_arr = []
    for i, value in np.ndenumerate(neural_data):
        # centering and / by std
        z_score = normalize_price(value, neural_train_data_mean, neural_train_data_std)
        minute = {
            "z_score": value,
            "std": std[index_in_norm_arr],
            'sec': time_sec[index_in_norm_arr],
            'minute': time_minute[index_in_norm_arr],
            'hour': time_hour[index_in_norm_arr],
            'timestamp': time_unix[index_in_norm_arr],
            'pseudo_log_return': pseudo_log_return_arr[index_in_norm_arr],
            'log_return': log_return_arr[index_in_norm_arr],
            'price_growth': price_growth_arr[index_in_norm_arr],
            'close_price': close_price_arr[index_in_norm_arr],
            'average_prices'
        }
        index_in_norm_arr += 1
        # insert minute in arr
        neural_data_mongo.insert_one(minute)

    print("DATA NORMALIZED in {} seconds".format(time.time() - unix_now))

def make_dataset(, test, **mean_statistics):
    collection = mongo.db["neural_train_data_" + index_name]
    collection.remove()
# def reject_outliers(data, m=2):
#     return data[abs(data - np.mean(data)) < m * np.std(data)]
