import flask_restful as restful
from server import mongo
from datetime import datetime
import numpy as np
import time
import math
from pprint import pprint
from sklearn.preprocessing import scale, MinMaxScaler, normalize, robust_scale, maxabs_scale, minmax_scale
from sklearn.decomposition import PCA

class Neural_Data(restful.Resource):
    # divide our finance collections in train, test datasets
    # Train = 70 %
    # Test = 30%

    def get(self):

        normalize_data_for_NN('GSPC')
        # normalize_data_for_NN('DJI')
        return "DATA STORED!"


def z_score(value, obj):
    return (value - obj["mean"]) / obj["std"]
#
# def toFixed(num):
#     return  "{:10.2f}".format(num)

def sk_min_max(X):
    min_max_scaler = MinMaxScaler()
    # X = scale(X, axis=0, with_mean=True, with_std=True, copy=True)
    return min_max_scaler.fit_transform(X)

def sk_normalize(X):
    return normalize(X, norm='l2')

def sk_scale(X):
    return scale(X, axis=0, with_mean=True, with_std=True, copy=True )

def sk_abs_scale(X):
    return maxabs_scale(X)

def sk_robust(X):
    return robust_scale(X)

def sk_pca(X):
    pca = PCA(n_components=3)
    return pca.fit(X)

def noop(x):
    return x

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
    # np.zeros(dataset_size)
    std_arr = [] #np.array([])  # std of prices on every minute
    time_sec = [] #np.array([])  # prices :ss on every minute
    time_hour = [] #np.array([])  # prices :HH on every minute
    time_minute = [] #np.array([])  # prices :MM on every minute
    time_unix = [] #np.array([])  # timestamp of every minute
    average_price_arr = [] #np.array([])  # average_price of every minute
    close_price_arr = [] #np.array([])  # average_price of every minute
    # last_data_timestamp = 1  # for DB queries, look at "data" variable
    last_average = 1  # last average price defined outside the loop scope
    last_close = 1  # last close price defined outside the loop scope
    log_return_arr = [] #np.array([])  # logarithmic difference of close prices of every consecutive minutes
    # same as log_return, but average price instead of close_price
    pseudo_log_return_arr = [] #np.array([])
    # price_growth_arr = []  # more about "price_growth" below
    price_growth_percent_arr = [] #np.array([])  # more about "price_growth_percent" below

    count = dataset_size
    # count = 100000
    while count > 0:
        # we can't get all data in one request, because we can have more than 400k
        # of records in every collection, so => limit(1000)
        limit = 150000
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
            # time_exclude = [1341815455, 1455021059, 1440153052, 1466764212, 1441103455, 1451910652, 1452861012, 1440498634, 1413455442, 1357111911, 1355902255, 1341815515, 1413887457, 1440585047, 1441794659, 1478521822, 1341815575]
            # if unix_timestamp in time_exclude or (unix_timestamp > 1440405000 and unix_timestamp < 1440436000):
            #     continue

            sec = int(local_time.strftime("%S"))
            minute = int(local_time.strftime("%M"))
            # we can count only active trading hours
            hour = int(local_time.strftime("%H")) - 13

            # if hour < 2 or hour > 5:
            #     continue

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
            # price growth compared to prev minute  normalized????

            pseudo_log_return = np.log(average / last_average)
            log_return = np.log(close_price / last_close)

            last_average = average
            last_close = close_price

            pseudo_log_return_arr = concat_timeline_value(pseudo_log_return_arr, pseudo_log_return)
            average_price_arr = concat_timeline_value(average_price_arr, average)
            close_price_arr = concat_timeline_value(close_price_arr, close_price)
            log_return_arr = concat_timeline_value(log_return_arr, log_return)
            time_minute = concat_timeline_value(time_minute, minute)
            time_hour = concat_timeline_value(time_hour, hour)
            time_sec = concat_timeline_value(time_sec, sec)
            time_unix = concat_timeline_value(time_unix, unix_timestamp)
            price_growth_percent_arr = concat_timeline_value(price_growth_percent_arr, price_growth_percent)
            std_arr = concat_timeline_value(std_arr, np.std(prices))

    # I'm not sure that this function can remove outliers
    # neural_data = reject_outliers(neural_data) # remove outliers

    cbs = [noop]
    price_growth_percent_scaled_arr = compose(price_growth_percent_arr, cbs)
    price_growth_percent_arr = compose(price_growth_percent_arr, cbs)
    average_price_arr = compose(average_price_arr, cbs)
    close_price_arr = compose(close_price_arr, cbs)
    pseudo_log_return_arr = compose(pseudo_log_return_arr, cbs)
    log_return_arr = compose(log_return_arr, cbs)
    std_arr = compose(std_arr, cbs)

    X = np.hstack(
    [average_price_arr, close_price_arr, log_return_arr, pseudo_log_return_arr, std_arr,  price_growth_percent_scaled_arr]
    )
    X = robust_scale(X)

    print("X (all data) array is ready.")
    store_mongo_arr = []
    arr_i = []
    for i in range(0, len(close_price_arr)):
    # for i in range(0, np.prod(close_price_arr.shape)):
        if X[i][5] > 2 or X[i][5] < -2:
            arr_i.append(i)
            continue

        if X[i][4] > 2 or X[i][4] < -2:
            arr_i.append(i)
            continue

        if X[i][3] > 2 or X[i][3] < -2:
            arr_i.append(i)
            continue

        store_mongo_arr.append(X[i])  # insert minute in arr
    print("X (no outliers data) array is ready.")
    for i, value in enumerate(arr_i):
        time_sec.remove(time_sec[i])
        time_minute.remove(time_minute[i])
        time_hour.remove(time_hour[i])
        time_unix.remove(time_unix[i])

    X_store = np.array(store_mongo_arr)
    store_mongo_arr = []
    X = minmax_scale(X_store, feature_range=(-1, 1))
    print("X (no outliers data + min/max scale) array is ready.")
    for i in range(0, len(time_sec)):
        minute = {
            "std": X[i][4],
            'sec': time_sec[i][0],
            'minute': time_minute[i][0],
            'hour': time_hour[i][0],
            'timestamp': time_unix[i][0],
            'pseudo_log_return': X[i][3],
            'log_return': X[i][2],
            'price_growth_percent_normalized': X[i][5],
            'price_growth_percent': price_growth_percent_arr[i][0],
            'close_price': X[i][1]
            # 'average_prices': X[i][0]
        }
        store_mongo_arr.append(minute)
        if len(store_mongo_arr) == 5000:
            collection.insert_many(store_mongo_arr)
            store_mongo_arr = []
    return None
# def reject_outliers(data, m=2):
#     return data[abs(data - np.mean(data)) < m * np.std(data)]
#
# /* 1 */
# {
#     "_id" : ObjectId("58c5671a6cabea457cb48be8"),
#     "minute" : 45.0,
#     "timestamp" : 1484678701.0,
#     "hour" : 7.0,
#     "log_return" : 0.0039602028343283,
#     "pseudo_log_return" : 0.00364310220153559,
#     "average_prices" : 0.985701148631159,
#     "close_price" : 0.988919730372908,
#     "price_growth_percent" : 2.08723535662629e-05,
#     "std" : 0.0114564600806767,
#     "sec" : 1.0
# }

def compose(x, arr):
    for func in arr:
        x = func(x)
    return x

def concat_timeline_value(arr, x):
    arr.append([x])
    return arr
    # return np.stack([arr, x])
