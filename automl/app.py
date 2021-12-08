import io
import os
import time
import json
import pickle
from uuid import uuid4
from threading import Thread
from flask import Flask, jsonify, request, send_file
from classification import ClassificationInputModel, do_autoclassification, NotSupportedMetricException

app = Flask(__name__)


def check_calculation_status_decorator(funciton):
    def wrapper(*args, **kwargs):
        model_id = request.args.get('model_id')

        if f'{model_id}.log' in os.listdir('error_logs'):
            with open(os.path.join('error_logs', f'{model_id}.log'), 'r') as file:
                exception = file.readline()
            return exception, 500

        elif f'{model_id}.pickle' not in os.listdir('saved_models'):
            return 'not calculated yet', 102

        else:
            return funciton(*args, **kwargs)
    wrapper.__name__ = funciton.__name__
    return wrapper


@app.route('/start_classification', methods=['POST'])
def auto_classification():
    try:
        model_id = uuid4()
        params = ClassificationInputModel(**json.loads(request.data))
        thread = Thread(target=do_autoclassification, args=(params, model_id))
        thread.start()
        return jsonify({'model_id': model_id}), 201
    except NotSupportedMetricException as e:
        return str(e), 400


@app.route('/get_model')
@check_calculation_status_decorator
def get_model():
    model_id = request.args.get('model_id')
    
    with open(os.path.join('saved_models', f'{model_id}.pickle'), 'rb') as file:
        model = pickle.load(file)
    return send_file(io.BytesIO(model._to_pickle_string()), download_name='model.pickle'), 200


@app.route('/get_score')
@check_calculation_status_decorator
def get_score():
    model_id = request.args.get('model_id')
    with open(os.path.join('saved_models', f'{model_id}.pickle'), 'rb') as file:
        model = pickle.load(file)

    print(model.score)
    return {'model_id': model_id, 'score': model.score, 'score_type': model.score_type}, 200

