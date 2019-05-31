import gensim
from gensim.models import Word2Vec
import multiprocessing
import logging
import os
import sys
import argparse
import yaml
from yaml import Loader

# allows import from skipgram-rnn directory
abspath_file = os.path.abspath(os.path.dirname(__file__))
skipgram_rnn_path = "/".join(abspath_file.split("/")[:-1])
sys.path.append(skipgram_rnn_path)

from tools.preprocessing import iterReviewsFile

# get environnement info
env = yaml.load(open(os.path.join(skipgram_rnn_path, "env.yml"), 'r'), Loader=Loader)


# API :
# https://radimrehurek.com/gensim/models/base_any2vec.html#gensim.models.base_any2vec.BaseWordEmbeddingsModel
# https://radimrehurek.com/gensim/models/word2vec.html
# interesting codes :
# https://github.com/kavgan/nlp-in-practice/blob/master/word2vec/Word2Vec.ipynb
# tutorial :
# rare technology word2vec


# ----------------------------------------------------------------------------

NB_REVIEWS = 50000
PROJECT_PATH = env["project_abspath"]
PATH_TO_QUESTIONS_WORDS_FILE = PROJECT_PATH + "models/questions-words.txt"

# model config
DEFAULT_STORE_MODEL_PATH = PROJECT_PATH + "/models"
DEFAULT_MODEL_NAME = "model_test"
MODEL_CONFIG_FILE = "config.yml"

#  ----------------------------------------------------------------------------


# class to create memory-friendly iterator over reviews
class MyReviews(object):
    def __init__(self, nb_reviews):
        self.nb_reviews = nb_reviews

    def __iter__(self):
        for i, filepath in enumerate(iterReviewsFile()):
            with open(file=filepath) as f:
                for line in f:
                    # do some pre-processing and return a list of words for each review text
                    yield gensim.utils.simple_preprocess(line)
            if i >= self.nb_reviews-1:
                break



def skipgram(init,
             load,
             model_path,
             model_name,
             save_kv,
             model_config,
             train,
             epochs,
             similarity,
             accuracy):
    """
        Arguments:
            init (bool) : initialize a skipgram model
            load (bool) : load a pre-trained model (or empty one)
            model_path (str) : path to the models directory
            model_name (str) : name of the model we want to use
    """

    # allows display info
    logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)


    # define some path variable to clean the code
    path_to_model_dir = os.path.join(model_path, model_name)
    path_to_model_file = os.path.join(path_to_model_dir, model_name+".model")
    path_to_keyed_vectors_file = os.path.join(path_to_model_dir, model_name+".kv")

    # use a memory-friendly iterator
    sentences = MyReviews(nb_reviews=NB_REVIEWS)

    if init and not load:
        # sentences / corpus = None so the model is left uninitialized
        # iter = 1 to make sure to have an uninitialized model
        model = Word2Vec(sentences=sentences,
                         iter=1,
                         size=model_config["size"],
                         window=model_config["window"],
                         min_count=model_config["min_count"],
                         hs=model_config["hs"],
                         negative=model_config["negative"],
                         workers=model_config["workers"])

        # save the model after initialization
        model.save(path_to_model_file)

    elif load:
        # load the model
        model = Word2Vec.load(path_to_model_file)

    else:
        # the user is informed that he has to choise init or load arguments
        raise RuntimeError("You have either to choose init or load")


    if train:
        # train the model
        model.train(sentences=sentences,
                    total_examples=model.corpus_count,
                    epochs=epochs)

        # always save the model after training
        model.save(path_to_model_file)


    if save_kv:
        # save vectors representation of words
        model.wv.save(path_to_keyed_vectors_file)

    if similarity != "":
        # evaluate the model by similarity search for one word
        print("Words similar to ", similarity)
        print(model.most_similar(positive=[similarity]))

    if accuracy:
        model.wv.accuracy(questions=PATH_TO_QUESTIONS_WORDS_FILE)

    # load vectors representation of words
    # wv = KeyedVectors.load("model.wv", mmap='r')



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-init", action="store_true")
    parser.add_argument("-load", action="store_true")
    parser.add_argument("-train", action="store_true")
    parser.add_argument("-model_path", type=str, default=DEFAULT_STORE_MODEL_PATH)
    parser.add_argument("-model_name", type=str, default=DEFAULT_MODEL_NAME)
    parser.add_argument("-save_kv", action="store_true")
    parser.add_argument("-epochs", type=int, default=1)
    parser.add_argument("-similarity", type=str, default="")
    parser.add_argument("-acc", action="store_true")

    args = parser.parse_args()

    # get model configuration
    stream = open(os.path.join(args.model_path, args.model_name, MODEL_CONFIG_FILE), 'r')
    model_config = yaml.load(stream, Loader=Loader)

    # execute skipgram
    skipgram(init=args.init,
             load=args.load,
             model_path=args.model_path,
             model_name=args.model_name,
             save_kv=args.save_kv,
             model_config=model_config,
             train=args.train,
             epochs=args.epochs,
             similarity=args.similarity,
             accuracy=args.acc)
