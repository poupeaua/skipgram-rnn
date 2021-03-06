import gensim
import os
import sys
import argparse
import yaml
from yaml import Loader
import numpy as np
from keras.models import Sequential, load_model
from keras.layers import Dense
from keras.layers import LSTM, CuDNNLSTM
from keras.layers import Dropout
from progress.bar import Bar

# allows import from skipgram-rnn directory
ABSPATH_FILE = os.path.abspath(os.path.dirname(__file__))
SKIPGRAM_RNN_PATH = "/".join(ABSPATH_FILE.split("/")[:-1])
sys.path.append(SKIPGRAM_RNN_PATH)

from tools.preprocessing import iter_reviews_as_model_inout, get_inout_from_review, iter_reviews_file

# get environment info
env = yaml.load(open(os.path.join(SKIPGRAM_RNN_PATH, "env.yml"), 'r'), Loader=Loader)

# ----------------------------------------------------------------------------

NB_REVIEWS = 50000
MAX_LENGTH_PREPROCESS_REVIEW = 5000
PROJECT_PATH = env["project_abspath"]

# rnn model default config
DEFAULT_RNN_STORE_MODEL_PATH = os.path.join(PROJECT_PATH, "models/rnn")
DEFAULT_RNN_MODEL_NAME = "model_test"
DEFAULT_TRAINING_SIZE = 25000
DEFAULT_TESTING_SIZE = 25000

# skipgram default config
DEFAULT_SKIPGRAM_STORE_MODEL_PATH = os.path.join(PROJECT_PATH, "models/skipgram/")
DEFAULT_SKIPGRAM_MODEL_NAME = "model_test"
SKIPGRAM_MODEL_CONFIG_FILE = "config.yml"

PATHS_TRAIN_DATA = [os.path.join(PROJECT_PATH, "data/aclImdb/train/pos"),
                    os.path.join(PROJECT_PATH, "data/aclImdb/train/neg")]
PATHS_TEST_DATA = [os.path.join(PROJECT_PATH, "data/aclImdb/test/pos"),
                   os.path.join(PROJECT_PATH, "data/aclImdb/test/neg")]


#  ----------------------------------------------------------------------------


# 82% classic embedding
# 22861 reviews with >= 80 words
def rnn(rnn_model_path,
        rnn_model_name,
        sg_model_path,
        sg_model_name,
        sg_model_config,
        init,
        train,
        load,
        test,
        pred,
        epochs,
        training_size,
        testing_size,
        gpu):
    """
        Arguments:
            init (bool) :
            train (bool) :
            load (bool) :
            test (bool) :
    """
    # path used for rnn model loading and saving and skipgram words embeddings
    path_to_rnn_model_file = os.path.join(rnn_model_path, rnn_model_name, rnn_model_name + ".h5")
    path_to_sg_model_kv_file = os.path.join(sg_model_path, sg_model_name, sg_model_name + ".kv")
    words_embeddings = gensim.models.KeyedVectors.load(path_to_sg_model_kv_file, mmap="r")
    embeddings_size = sg_model_config["size"]

    if init and not load:
        print('Build model...')
        model = Sequential()
        print("GPU", gpu)
        if gpu == 0:
            # as the input shape is (None, embeddings_size) => DYNAMIC RNN
            model.add(LSTM(128, dropout=0.2, recurrent_dropout=0.2, input_shape=(None, embeddings_size)))
        else:
            # as the input shape is (None, embeddings_size) => dynamic RNN
            # CuDNNLSTM fast version of LSTM (can only be run with GPU)
            model.add(CuDNNLSTM(128, input_shape=(None, embeddings_size)))
            model.add(Dropout(0.2))
        model.add(Dense(1, activation='sigmoid'))

        # try using different optimizers and different optimizer configs
        model.compile(loss='binary_crossentropy',
                      optimizer='adam',
                      metrics=['accuracy'])

        # save model after initializing it
        print("Save model...")
        model.save(path_to_rnn_model_file)

    elif load:
        # load model from file
        model = load_model(filepath=path_to_rnn_model_file)

    else:
        raise RuntimeError("You have either to choose parameter -init or -load")

    if train:
        print('Train...')

        training_iterator = iter_reviews_as_model_inout(words_embeddings=words_embeddings,
                                                        paths=PATHS_TRAIN_DATA,
                                                        max_nb_reviews=training_size)

        # DYNAMIC RNN with nb_words = None
        for cur_epoch in range(epochs):
            for input, label in Bar('Processing epoch ' + str(cur_epoch), max=training_size).iter(training_iterator):
                model.fit(np.array([input]), [label], batch_size=1, epochs=1, verbose=0)

        # save model after training
        print("Save model...")
        model.save(path_to_rnn_model_file)

    if test:
        print('Test...')

        testing_iterator = iter_reviews_as_model_inout(words_embeddings=words_embeddings,
                                                       paths=PATHS_TEST_DATA,
                                                       max_nb_reviews=testing_size)
        confusion_matrix = np.zeros(shape=(2, 2))
        for input, label in Bar('Processing', max=testing_size).iter(testing_iterator):
            prediction = model.predict_classes(x=np.array([input]))
            confusion_matrix[label, prediction] += 1
        print("Confusion matrix :")
        print(confusion_matrix)
        print('Test accuracy:', np.sum(np.trace(confusion_matrix)) / testing_size)

    if pred:
        all_review_paths = list(iter_reviews_file(paths=PATHS_TEST_DATA))
        random_review_path = np.random.choice(all_review_paths)
        print("Review :", random_review_path)
        with open(file=random_review_path, mode='r') as f:
            print("Random review used for prediction")
            print(f.read())
        input, label = get_inout_from_review(words_embeddings=words_embeddings,
                                             review_path=random_review_path)
        print("True label :", label)
        print("Prediction label :", model.predict_classes(x=np.array([input]))[0][0])
        print("Prediction :", model.predict_proba(x=np.array([input]))[0][0])


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-init", action="store_true")
    parser.add_argument("-load", action="store_true")
    parser.add_argument("-train", action="store_true")
    parser.add_argument("-test", action="store_true")
    parser.add_argument("-pred", action="store_true")
    parser.add_argument("-sg_model_path", type=str, default=DEFAULT_SKIPGRAM_STORE_MODEL_PATH)
    parser.add_argument("-sg_model_name", type=str, default=DEFAULT_SKIPGRAM_MODEL_NAME)
    parser.add_argument("-rnn_model_path", type=str, default=DEFAULT_RNN_STORE_MODEL_PATH)
    parser.add_argument("-rnn_model_name", type=str, default=DEFAULT_RNN_MODEL_NAME)
    parser.add_argument("-epochs", type=int, default=1)
    parser.add_argument("-training_size", type=int, default=DEFAULT_TRAINING_SIZE)
    parser.add_argument("-testing_size", type=int, default=DEFAULT_TESTING_SIZE)
    parser.add_argument("-gpu", type=int, default=0)

    args = parser.parse_args()

    # get model configuration
    stream = open(os.path.join(args.sg_model_path, args.sg_model_name, SKIPGRAM_MODEL_CONFIG_FILE), 'r')
    sg_model_config = yaml.load(stream, Loader=Loader)

    # execute rnn
    rnn(init=args.init,
        load=args.load,
        rnn_model_path=args.rnn_model_path,
        rnn_model_name=args.rnn_model_name,
        sg_model_path=args.sg_model_path,
        sg_model_name=args.sg_model_name,
        sg_model_config=sg_model_config,
        train=args.train,
        test=args.test,
        pred=args.pred,
        epochs=args.epochs,
        training_size=args.training_size,
        testing_size=args.testing_size,
        gpu=args.gpu)
