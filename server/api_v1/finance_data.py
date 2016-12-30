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
        finance_data = mongo.db.finance_data
        data = finance_data.find({}).limit(100)
        dicti = {}
        for x in data:
            timestamp = x["timestamp"]
            price = x["close_price"]
            dicti[timestamp] = price
        open('lol.json', 'a').close()
        file_d = open('lol.json', 'r+')
        file_d.write(json.dumps(dicti))
        file_d.close()

#         with open('./data/GSPC_1m.csv', 'r') as file_data:
#             for line in file_data:
#                 if i == 0:
#                     # hardcode, skip header
#                     i = 1
#                     continue
#                 row = line.split(';')
#                 timestamp = datetime.strptime(row[2], '%Y-%m-%d %H:%M:%S').timestamp()
#                 open_price = to_float(row[3])
#                 high_price = to_float(row[4])
#                 low_price = to_float(row[5])
#                 close_price = to_float(row[6])
#                 volume = to_float(row[7])
#
#                 minute_data = Minute_data(open_price=open_price, timestamp=timestamp, volume=volume, low_price=low_price, close_price=close_price, high_price=high_price)

#                 finance_data.insert_one(minute_data.__dict__)
#
# class Minute_data:
#     def __init__(self, **kwargs):
#         for key, value in kwargs.items():
#             setattr(self, key, value)
