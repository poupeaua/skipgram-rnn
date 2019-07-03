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
        curr_res = read_results(model_path=os.path.join(MODEL_PATH, md), model_name=md)
        if curr_res['model_name'] not in models_res:
            models_res[curr_res['model_name']] = curr_res['metrics']
    return models_res


# Function that allows to plot confusion matrix

# IMPORTANT: this function has been extracted from open source code in the internet, and modified to receive the correct
# parameters.
# reference: https://scikit-learn.org/stable/auto_examples/model_selection/plot_confusion_matrix.html

def plot_confusion_matrix(matrix, classes_list, normalize=False, title=None, cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    if not title:
        if normalize:
            title = 'Normalized confusion matrix'
        else:
            title = 'Confusion matrix, without normalization'

    # Compute confusion matrix
    cm = matrix
    # Only use the labels that appear in the data
    # classes_list = classes_list[unique_labels(y_true, y_pred)]
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    print(cm)

    fig, ax = plt.subplots()
    im = ax.imshow(cm, interpolation='nearest', cmap=cmap)
    ax.figure.colorbar(im, ax=ax)
    # We want to show all ticks...
    ax.set(xticks=np.arange(cm.shape[1]),
           yticks=np.arange(cm.shape[0]),
           # ... and label them with the respective list entries
           xticklabels=classes_list, yticklabels=classes_list,
           title=title,
           ylabel='Etiqueta Correcta',
           xlabel='Etiqueta Predicha')

    # Rotate the tick labels and set their alignment.
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right",
             rotation_mode="anchor")

    # Loop over data dimensions and create text annotations.
    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, format(cm[i, j], fmt),
                    ha="center", va="center",
                    color="white" if cm[i, j] > thresh else "black")
    fig.tight_layout()
    return ax


results = get_results()

windows = [2, 3, 5, 10]
sizes = [10, 25, 50, 100, 200, 500]

best_cm = results['model_500_10']['cm']

plot_confusion_matrix(best_cm, ['Mala', 'Buena'], normalize=True, title='Matriz de Confusión para Modelo M500-10')
plt.savefig('cm.png')
plt.show()

for w in windows:
    acc = []
    for s in sizes:
        model = 'model_{}_{}'.format(s, w)
        acc.append(results[model]['accuracy'])
    plt.plot(sizes, acc, label='window size = {}'.format(w), marker='.', linewidth='1')
    print(acc)

plt.xlabel('Tamaño de los Vectores de Palabras')
plt.ylabel('Precisión')
plt.title('Gráfico de Presición v/s Tamaño de Vector\npara Distintos Tamaños de Ventanas')
plt.grid(True)
plt.legend()
plt.savefig('plot.png')
plt.show()
