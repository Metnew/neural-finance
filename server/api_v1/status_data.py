import json
from flask import request, jsonify, g
import flask_restful as restful
from server import mongo
from datetime import timedelta, datetime

class Status_Data(restful.Resource):
    def get(self):
        status_table = mongo.db.status_table
        current = status_table.find_one().sort('timestamp', -1).limit(1)
        return current
