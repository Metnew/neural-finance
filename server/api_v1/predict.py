import json
from flask import request, jsonify, g
import flask_restful as restful
from server import mongo
from datetime import timedelta, datetime
import numpy as np
from clint.textui import colored
import requests
from keras.models import model_from_json, load_model
from .neural_data  import Neural_Data

class Prediction(restful.Resource):
    def get(self):
        batch_size = 1
        rolling_window_size = 5
        # r = requests.get('http://localhost:5000/api/v1/neural_data') # Call neural_data
        # print(r)
        Neural_Data.get(Neural_Data)
        data = list(mongo.db.neural_data.find({}).sort('timestamp', -1).limit(5))
        arr = create_rolling_window(data, rolling_window_size)
        X = np.array([arr])
        X = np.reshape(X, (X.shape[0], 1, X.shape[1]))

        json_file = open('model.json', 'r')
        loaded_model_json = json_file.read()
        json_file.close()
        model = model_from_json(loaded_model_json)
        # load weights into new model
        model.load_weights("model.h5")
        print("Loaded model from disk")

        prediction = model.predict(X, batch_size=batch_size, verbose=1)
        prediction = prediction.tolist()[0][0]
        print(colored.green("Make new prediction...And its {}".format(prediction)))
        prediction = round(prediction, 0)
        result = {
            'price_growth': True
        }
        if prediction == 0:
            result["price_growth"] = False
            print('Price will fall')
        else:
            print('Price will growth')

        return jsonify(result)

def create_rolling_window(data, size):
    arr = []
    i = 5
    for inner in range(1, size + 1):
        # minute_score = data[i - inner]["z_score"]
        std = data[i - inner]["std"]
        price_growth = data[i - inner]["price_growth"]
        pseudo_log_return = data[i - inner]["pseudo_log_return"] * 1000
        log_return = data[i - inner]["log_return"] * 1000
        arr.append(log_return)
        arr.append(pseudo_log_return)
        arr.append(std)
        arr.append(price_growth)
    return arr
