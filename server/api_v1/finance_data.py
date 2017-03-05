import json
from flask import request, jsonify, g
import flask_restful as restful
from server import mongo
from datetime import timedelta, datetime
import numpy as np
import csv


def to_float(a):
    return float(a.replace(',', '.'))


class Finance_Data(restful.Resource):

    def get(self):
        # store data in mongo
        def store_in_mongo(name, *url):
            finance_data = mongo.db['finance_data_' + name]# get collection e.g. finance_data_GSPC
            finance_data.remove() # remove collection if already exists
            arr = []  # temp data storing array
            if not url:
                url = './csv/' + name + '.csv'
            with open(url, 'r') as file_data:
                reader = csv.reader(file_data, delimiter=";")
                next(reader, None)
                for row in reader:
                    timestamp = datetime.strptime(row[2], '%Y-%m-%d %H:%M:%S').timestamp()
                    open_price = to_float(row[3])
                    high_price = to_float(row[4])
                    low_price = to_float(row[5])
                    close_price = to_float(row[6])
                    volume = to_float(row[7])

                    minute_data = Minute_data(open_price=open_price, timestamp=timestamp, volume=volume, low_price=low_price, close_price=close_price, high_price=high_price)
                    # append data to array
                    arr.append(minute_data.__dict__)
                    # if array size bigger then 1000 store array in mongo
                    # we cant store minutes in mongo one by one - it will be too slow
                    if len(arr) == 1000:
                        finance_data.insert_many(arr)
                        arr = []

        store_in_mongo(name="GSPC")
        store_in_mongo(name="DJI")


class Minute_data:

    def __init__(self, **kwargs):
        for key, value in kwargs.items():
            setattr(self, key, value)
