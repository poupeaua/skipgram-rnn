import os
import numpy as np
import matplotlib.pyplot as plt

# allows import from skipgram-rnn directory
ABS_PATH = os.path.split(os.path.abspath(os.path.dirname(__file__)))[0]
MODEL_PATH = os.path.join(ABS_PATH, 'models/rnn')
MODELS_DIRS = os.listdir(MODEL_PATH)


def read_results(model_path, model_name=''):
    res = {
        'model_name': model_name
    }
    for f in os.listdir(model_path):
        if f[-4:] == '.txt':
            with open(os.path.join(model_path, f), 'r') as res_file:
                results = res_file.read()
                split = results.split('\n')
                row1 = split[1].replace(' ', '')[2:-2].split('.')
                row2 = split[2].replace(' ', '')[1:-3].split('.')
                matrix = np.array([row1, row2], dtype=np.int)
                accuracy = float(split[-1].split(':')[1])
                metrics = {
                    'cm': matrix,
                    'accuracy': accuracy
                }
                res['metrics'] = metrics
    return res

def get_results():
    models_res = {}
    for md in MODELS_DIRS:
        print(md)
        curr_res = read_results(model_path=os.path.join(MODEL_PATH, md), model_name=md)
        if curr_res['model_name'] not in models_res:
            models_res[curr_res['model_name']] = curr_res['metrics']
    return models_res

