from flask import Flask, jsonify, request, current_app
from flask_pymongo import PyMongo
import flask_restful as restful
# import schedule
import time
# from threading import Thread

app = Flask(__name__)
app.config['MONGO_DBNAME'] = 'neural_finance'
app.config['MONGO_URI'] = 'mongodb://localhost:27017/neural_finance'
mongo = PyMongo(app)


from .api_v1.finance_data import Finance_Data
from .api_v1.neural_data import Neural_Data
# Status_Data - part of API implemented for model backtesting
# from .api_v1.status_data import Status_Data
from .api_v1.predict import Prediction
from .api_v1.train import Train

api = restful.Api(app, prefix='/api/v1')

api.add_resource(Train, '/train')
api.add_resource(Prediction, '/predict')
# api.add_resource(Status_Data, '/status_data')
api.add_resource(Finance_Data, '/finance_data')
api.add_resource(Neural_Data, '/neural_data')

# run tasks every minute
# from .jobs.tasks import accomplish_tasks

# def run_schedule():
#     while True:
#         schedule.run_pending()
#         time.sleep(1)
#
# schedule.every(1).minute.do(accomplish_tasks)
# t = Thread(target=run_schedule)
# t.start()



@app.errorhandler(403)
def forbidden_page(error):
    return jsonify({'error': 'forbidden'}), 403


@app.errorhandler(404)
def page_not_found(error):
    return jsonify({'error': 'not found'}), 404

@app.errorhandler(500)
def server_error_page(error):
    db.session.rollback()
    return jsonify({'error': 'internal server error'}), 500

# logger
# @app.before_request
# def log_request_info():
    # app.logger.debug('Headers: %s', request.headers)
    # app.logger.debug('Body: %s', request.get_data())
