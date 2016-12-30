import json
from flask import request, jsonify, g
import flask_restful as restful
from server import mongo
from datetime import timedelta, datetime
import numpy as np
import time

class Neural_Data(restful.Resource):
    def get(self):
        finance_data = mongo.db.finance_data
        neural_data = []
        neural_data_mongo = mongo.db.neural_data
        max_log_return = -1
        min_log_return = 111111111
        maximum = 0
        minimum = 0
        start_date = 1468920744
        std = []
        time_sec = []
        time_hour = []
        time_minute = []
        time_unix = []
        last_average = 1
        average_prices = []
        last_close = 1
        pseudo_log_return_arr = []
        log_return_arr = []
        stringed_time = []
        price_growth_arr = []
        unix_now = time.time()
        # selected data
        # for i in range(0, 30000):
        print('START NORMALIZE DATA')
        data = finance_data.find({"timestamp": {"$lte": int(unix_now)}}).limit(50000)
        # useless code for timestamp normalization
        # for minute in data:
        #     start_date = minute["timestamp"]
        #     end = start_date % 10
        #     if end > 4 :
        #         end = 10 - end
        #     else:
        #         end -= end
        # finance_data.update({}, {"$inc":{"timestamp":end}})
        # data = finance_data.find({"timestamp": {"$gt": int(start_date)}}).limit(30000)
            # prev_index_1 = i - 1
            # prev_index_2 = i - 2
            # prev_index_3 = i - 3
            # prev_index_4 = i - 4
            # prev_index_5 = i - 5
            # prev_index_6 = i - 6
            # min_max_scaler = preprocessing.MinMaxScaler()
            # current_price = close_price[i]
            # prev_price_1 = get_price()minute[prev_index_1]
            # prev_price_2 = close_price[prev_index_2]
            # prev_price_3 = close_price[prev_index_3]
            # prev_price_4 = close_price[prev_index_4]
            # prev_price_5 = close_price[prev_index_5]
            # prev_price_6 = close_price[prev_index_6]
            #
            # arr_current_minute = np.array(
            #     [open_price[i], close_price[i], high_price[i], low_price[i]])
            # arr_last_minute_1 = np.array([open_price[prev_index_1], close_price[
            #                              prev_index_1], high_price[prev_index_1], low_price[prev_index_1]])
            # arr_last_minute_2 = np.array([open_price[prev_index_2], close_price[
            #                              prev_index_2], high_price[prev_index_2], low_price[prev_index_2]])
            # arr_last_minute_3 = np.array([open_price[prev_index_3], close_price[
            #                              prev_index_3], high_price[prev_index_3], low_price[prev_index_3]])
            # arr_last_minute_4 = np.array([open_price[prev_index_4], close_price[
            #                              prev_index_4], high_price[prev_index_4], low_price[prev_index_4]])
            # arr_last_minute_5 = np.array([open_price[prev_index_5], close_price[
            #                              prev_index_5], high_price[prev_index_5], low_price[prev_index_5]])
            # arr_last_minute_6 = np.array([open_price[prev_index_6], close_price[
            #                              prev_index_6], high_price[prev_index_6], low_price[prev_index_6]])
        # regr = linear_model.LinearRegression()
        for x in data:
            minute = x

            # print(minute)
            open_price = minute["open_price"]
            close_price = minute["close_price"]
            low_price = minute["low_price"]
            high_price = minute["high_price"]
            start_date = minute["timestamp"]
            prices = np.array([open_price, low_price, close_price, high_price])
            # last_close = close_price
            average = np.average(prices)
            average_prices.append(average)

            local_time = datetime.fromtimestamp(start_date)
            # stringed_time.append(local_time.strftime('%Y-%m-%d %H:%M:%S'))
            time_unix.append(start_date)
            time_sec.append(int(local_time.strftime("%S")))
            time_minute.append(int(local_time.strftime("%M")))
            time_hour.append(int(local_time.strftime("%H")) - 13)

            # print(len(range(0, len(average_prices) )))
            # print(len(average_prices))
            # regr.fit(np.array(range(0, len(average_prices) + 1)).reshape(len(average_prices), 1), np.array(average_prices).reshape(len(average_prices) , 1))
            # print(regr.coef_)
            if close_price - last_close > 0 :
                price_growth = 1
            elif close_price - last_close < 0 :
                price_growth = 0
            price_growth_arr.append(price_growth)
            pseudo_log_return = np.log(average / last_average)
            pseudo_log_return_arr.append(pseudo_log_return)

            log_return = np.log(close_price / last_close)
            log_return_arr.append(log_return)

            # dates = pd.date_range(stringed_time[0], stringed_time[-1], freq="m")
            # AO = pd.Series(data=log_return_arr, index = dates)
            # z = AO.reset_index()
            # try:
            #     model = pd.ols(x=pd.to_datetime(z["index"]).dt.minute, y=z[0])
            #     print(model)
            # except Exception as e:
            #     print(e)
            last_average = average
            last_close = close_price
            neural_data.append(average)
            std.append(np.std(prices))
                # normalized_average = normalize_price(average)
            # neural_minute = {
            #     'timestamp': timestamp,
            #     'sec': time_sec,
            #     'minute': time_minute,
            #     'hour': time_hour,
            #     'std': std,
            #     'log_return':
            # }
            # neural_data.insert_one({}, neural_minute)
        neural_data = np.array(neural_data)
        neural_data = reject_outliers(neural_data)
        neural_data_mean = np.average(neural_data)
        neural_data_std = np.std(neural_data)

        index_in_norm_arr = 0

        neural_data_mongo.remove()
        for i, value in np.ndenumerate(neural_data):
            # print(value)
            # if minimum > value:
            #     minimum = value
            # if maximum < value:
            #     maximum = value
            value = normalize_price(value, neural_data_mean, neural_data_std)
            minute = {
                "z_score": value,
                "std": std[index_in_norm_arr],
                'sec': time_sec[index_in_norm_arr],
                'minute': time_minute[index_in_norm_arr],
                'hour': time_hour[index_in_norm_arr],
                'timestamp': time_unix[index_in_norm_arr],
                'pseudo_log_return': pseudo_log_return_arr[index_in_norm_arr],
                'log_return': log_return_arr[index_in_norm_arr],
                'price_growth': price_growth_arr[index_in_norm_arr]

            }
            index_in_norm_arr += 1
            neural_data_mongo.insert_one(minute)
        print("DATA NORMALIZED in {} seconds".format(time.time() - unix_now))
        return "DATA STORED!"


# def normalize_price(price, minimum, maximum):
def normalize_price(value, mean, std):
    # return ((2*price - (maximum + minimum)) / (maximum - minimum))
    return (value - mean) / std

def reject_outliers(data, m=2):
    return data[abs(data - np.mean(data)) < m * np.std(data)]
