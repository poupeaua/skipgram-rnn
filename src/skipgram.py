from gensim.test.utils import common_texts, get_tmpfile
from gensim.models import Word2Vec
import multiprocessing
import logging
import os
import sys
import argparse
import yaml
from yaml import Loader

abspath_file = os.path.abspath(os.path.dirname(__file__))
skipgram_rnn_path = "/".join(abspath_file.split("/")[:-1])
sys.path.append(skipgram_rnn_path)

from tools.preprocessing import loadReviewsNpy

# API :
# https://radimrehurek.com/gensim/models/base_any2vec.html#gensim.models.base_any2vec.BaseWordEmbeddingsModel
# https://radimrehurek.com/gensim/models/word2vec.html
# interesting codes :
# https://github.com/kavgan/nlp-in-practice/blob/master/word2vec/Word2Vec.ipynb

cpu_count = multiprocessing.cpu_count()


# ----------------------------------------------------------------------------

# model config
DEFAULT_STORE_MODEL_PATH = "/u/a/2019/apoupeau/Documentos/recpatr/skipgram-rnn/models"
DEFAULT_MODEL_NAME = "model_test"
MODEL_CONFIG_FILE = "config.yml"

# data config
PATH_STORE_REVIEWS_AS_ARRAYS = "/u/a/2019/apoupeau/Documentos/recpatr/skipgram-rnn/data/reviews_as_arrays/"
DEFAULT_STORE_FILENAME = "reviews_as_arrays.npy"

#  ----------------------------------------------------------------------------


def skipgram(init,
             load,
             model_path,
             model_name,
             save_embeddings,
             model_config,
             train,
             epochs):

    # define some path variable to clean the code
    path_to_model_dir = os.path.join(model_path, model_name)
    path_to_model_file = os.path.join(path_to_model_dir, model_name+".model")
    path_to_keyed_vectors_file = os.path.join(path_to_model_dir, model_name+".kv")

    if init and not load:
        # sentences / corpus = None so the model is left uninitialized
        model = Word2Vec(None,
                         size=model_config["size"],
                         window=model_config["window"],
                         min_count=model_config["min_count"],
                         hs=model_config["hs"],
                         negative=model_config["negative"],
                         workers=32)

        # save the model after initialization
        model.save(path_to_model_file)

    elif load:
        # load the model
        model = Word2Vec.load(path_to_model_file)

    else:
        # the user is informed that he has to choise init or load arguments
        raise RuntimeError("You have either to choose init or load")


    # train
    if train:
        s = loadReviewsNpy()
        print(s.shape)
        # for i in range(len(s)):
        #     s[i] = list(s[i])
        # print(s)
        model.train(sentences=s,
                    epochs=epochs)

        # always save the model after training
        model.save(path_to_model_file)


    if save_embeddings:
        # save vectors representation of words
        model.wv.save(path_to_keyed_vectors_file)

    # load vectors representation of words
    # wv = KeyedVectors.load("model.wv", mmap='r')



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-init", action="store_true")
    parser.add_argument("-load", action="store_true")
    parser.add_argument("-train", action="store_true")
    parser.add_argument("-model_path", type=str, default=DEFAULT_STORE_MODEL_PATH)
    parser.add_argument("-model_name", type=str, default=DEFAULT_MODEL_NAME)
    parser.add_argument("-save_embeddings", action="store_true")
    parser.add_argument("-epochs", type=int, default=1)

    args = parser.parse_args()

    # get model configuration
    stream = open(os.path.join(args.model_path, args.model_name, MODEL_CONFIG_FILE), 'r')
    model_config = yaml.load(stream, Loader=Loader)

    # execute skipgram
    skipgram(init=args.init,
             load=args.load,
             model_path=args.model_path,
             model_name=args.model_name,
             save_embeddings=args.save_embeddings,
             model_config=model_config,
             train=args.train,
             epochs=args.epochs)
