import gensim
from gensim.models import KeyedVectors
import logging
import os
import sys
import argparse
import yaml
from yaml import Loader
import keras
from keras.preprocessing import sequence
from keras.models import Sequential
from keras.layers import Dense, Embedding
from keras.layers import LSTM
from keras.datasets import imdb

# allows import from skipgram-rnn directory
abspath_file = os.path.abspath(os.path.dirname(__file__))
skipgram_rnn_path = "/".join(abspath_file.split("/")[:-1])
sys.path.append(skipgram_rnn_path)

from tools.preprocessing import iter_reviews_file


# get environnement info
env = yaml.load(open(os.path.join(skipgram_rnn_path, "env.yml"), 'r'), Loader=Loader)


# ----------------------------------------------------------------------------

NB_REVIEWS = 50000
MAX_LENGTH_PREPROCESS_REVIEW = 5000
PROJECT_PATH = env["project_abspath"]

# rnn model default config
DEFAULT_STORE_RNN_MODEL_PATH = os.path.join(PROJECT_PATH, "/models/rnn")
DEFAULT_RNN_MODEL_NAME = "model_test"
MODEL_CONFIG_FILE = "config.yml"

# skipgram default config
DEFAULT_SKIPGRAM_STORE_MODEL_PATH = os.path.join(PROJECT_PATH, "models/skipgram/")
PATH_TO_QUESTIONS_WORDS_FILE = os.path.join(DEFAULT_STORE_MODEL_PATH, "questions-words.txt")
DEFAULT_SKIPGRAM_MODEL_NAME = "model_test"
SKIPGRAM_MODEL_CONFIG_FILE = "config.yml"

#  ----------------------------------------------------------------------------


def rnn(rnn_model_path,
        rnn_model_name,
        init,
        train,
        load,
        test):
    """
        Arguments:
            init (bool) :
            train (bool) :
            load (bool) :
            test (bool) :
    """
    max_features = 20000
    # cut texts after this number of words (among top max_features most common words)
    maxlen = 80
    batch_size = 32

    print('Loading data...')
    (x_train, y_train), (x_test, y_test) = imdb.load_data(num_words=max_features)
    print(len(x_train), 'train sequences')
    print(len(x_test), 'test sequences')

    print('Pad sequences (samples x time)')
    x_train = sequence.pad_sequences(x_train, maxlen=maxlen)
    x_test = sequence.pad_sequences(x_test, maxlen=maxlen)
    print('x_train shape:', x_train.shape)
    print('x_test shape:', x_test.shape)

    print('Build model...')
    model = Sequential()
    model.add(Embedding(max_features, 128))
    model.add(LSTM(128, dropout=0.2, recurrent_dropout=0.2))
    model.add(Dense(1, activation='sigmoid'))

    # try using different optimizers and different optimizer configs
    model.compile(loss='binary_crossentropy',
                  optimizer='adam',
                  metrics=['accuracy'])

    print('Train...')
    model.fit(x_train, y_train,
              batch_size=batch_size,
              epochs=15,
              validation_data=(x_test, y_test))
    score, acc = model.evaluate(x_test, y_test,
                                batch_size=batch_size)
    print('Test score:', score)
    print('Test accuracy:', acc)
