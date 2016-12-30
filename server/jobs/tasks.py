import requests
import time
import schedule
import json
from clint.textui import colored
from datetime import timedelta, datetime
from server import mongo, app
import math

def accomplish_tasks():
    print(colored.yellow("Start tasks!"))
    print(colored.green("Get Data from Yahoo."))
    last_time = 0
    can_trade = False
    price_growth = True

    def get_data_by_schedule():
        test_env = True
        if test_env:
            return
        params = {
            'range': '1d',
            'interval': '1m',
            'includePrePost': False,
            'includeTimestamps': True,
            'indicators': "quote"
        }
        r = requests.get(
            'https://query2.finance.yahoo.com/v7/finance/chart/^GSPC', params)
        print(colored.green("Data Loaded."))
        data = r.json()
        result = data[u'chart'][u'result'][0]
        quote = result[u'indicators'][u'quote'][0]
        close_price = quote[u'close']
        open_price = quote[u'open']
        low_price = quote[u'low']
        high_price = quote[u'high']
        volume = quote[u'volume']
        timestamp = result[u'timestamp']
        finance_data = mongo.db.finance_data
        last_time = 0

        for i, time in enumerate(timestamp):
            if time < last_time:
                continue
            else:
                last_time = time
            exist = finance_data.find_one({'timestamp': time})
            if not exist:
                print(colored.cyan("Data for {} is loaded in DB.".format(
                    datetime.fromtimestamp(time).strftime("%Y-%m-%d %H:%M:%S"))))
                minute_data = {
                    'timestamp': time,
                    'close_price': close_price[i],
                    'open_price': open_price[i],
                    'low_price': low_price[i],
                    'high_price': high_price[i],
                    'volume': volume[i]
                }
                finance_data.insert_one(minute_data)

        return 1

    def make_prediction():
        r = requests.get('http://localhost:5000/api/v1/predict')
        print(r.json())
        return r.json()["price_growth"]
        # return result["price_growth"]


    def trade():
        status_data = mongo.db.status_data
        timestamp = rounded_to_minutes()
        formatted = datetime.fromtimestamp(timestamp).strftime("%Y-%m-%d %H:%M:%S")
        status = status_data.find_one({"timestamp": {"$lte": timestamp}})
        print(colored.cyan("START TRADE (IF POSSIBLE) AT {}!".format(formatted)))
        if not status:
            print(colored.green("Create new status data with 100k$!"))
            status = {
                "timestamp":  timestamp,
                "indexes": 0,
                "balance": 100000
            }
            status_data.insert_one(status)

        price_growth = make_prediction()
        if can_trade:
            print(colored.green("CAN Trade!"))
            balance = status["balance"]
            indexes = status["indexes"]
            if price_growth:
                indexes += math.floor(balance / current_price)
                balance -= indexes * current_price
            elif not price_growth:
                balance += indexes * current_price
                indexes = 0

            status_new = {
                'balance': balance,
                'indexes': indexes,
                'timestamp': timestamp
            }
            status_data.insert_one(status_new)
        else:
            print(colored.red("CANT Trade!"))


    def rounded_to_minutes():
        timestamp = time.time()
        timestamp -= 60 - timestamp % 60
        return timestamp


    with app.app_context():
        get_data_by_schedule()
        trade()
        #
        # colored.green("\n Train Model on new batch of data.")
        # colored.green("\n Get new prediction.")
        # colored.green("\n Send prediction")
        # Increment the minute total
