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

from tools.preprocessing import iter_reviews_file
from tools.subsampling import subsample

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

# model config
DEFAULT_SKIPGRAM_STORE_MODEL_PATH = os.path.join(PROJECT_PATH, "models/skipgram/")
PATH_TO_QUESTIONS_WORDS_FILE = os.path.join(DEFAULT_SKIPGRAM_STORE_MODEL_PATH, "questions-words.txt")
DEFAULT_SKIPGRAM_MODEL_NAME = "model_test"
SKIPGRAM_MODEL_CONFIG_FILE = "config.yml"

VOCABULARY_DICT_PATH = ""


#  ----------------------------------------------------------------------------


# class to create memory-friendly iterator over reviews
class MyReviews(object):
    def __init__(self, nb_reviews):
        self.nb_reviews = nb_reviews

    def __iter__(self):
        for i, filepath in enumerate(iter_reviews_file()):
            with open(file=filepath, encoding='utf-8') as f:
                # do some pre-processing and return a list of words for each review text
                tokenized_review = gensim.utils.simple_preprocess(f.read())
                yield tokenized_review
            if i >= self.nb_reviews - 1:
                break


def skipgram(init,
             load,
             sg_model_path,
             sg_model_name,
             save_kv,
             sg_model_config,
             train,
             epochs,
             similarity,
             accuracy):
    """
        Function used to handle the SkipGram model and generate the word
        embeddings.

        Arguments:
            init (bool) : initialize a skipgram model
            load (bool) : load a pre-trained model (or empty one)
            sg_model_path (str) : path to the skipgram models directory
            sg_model_name (str) : name of the skipgram model we want to use
            save_kv (bool) : save embeddings in a file .kv format in
            sg_model_config (dict) : configuration of the skipgram model
            train (bool) : active training
            epochs (int) : number of epochs for training
            similarity (str) : display ten words most similar to this word
            accuracy (bool) : active model evaluation
    """

    # allows display info
    logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)

    # define some path variable to clean the code
    path_to_model_dir = os.path.join(sg_model_path, sg_model_name)
    path_to_model_file = os.path.join(path_to_model_dir, sg_model_name + ".model")
    path_to_keyed_vectors_file = os.path.join(path_to_model_dir, sg_model_name + ".kv")

    # use a memory-friendly iterator
    sentences = MyReviews(nb_reviews=NB_REVIEWS)

    if init and not load:
        # sentences / corpus = None so the model is left uninitialized
        # iter = 1 to make sure to have an uninitialized model
        # sample = The threshold for configuring which higher-frequency words are randomly downsampled, useful range is (0, 1e-5).
        model = Word2Vec(sentences=sentences,
                         sg=1,
                         iter=1,
                         size=sg_model_config["size"],
                         window=sg_model_config["window"],
                         sample=sg_model_config["sample"],
                         min_count=sg_model_config["min_count"],
                         hs=sg_model_config["hs"],
                         negative=sg_model_config["negative"],
                         workers=sg_model_config["workers"])

        # save the model after initialization
        model.save(path_to_model_file)

    elif load:
        # load the model
        model = Word2Vec.load(path_to_model_file)

    else:
        # the user is informed that he has to choise init or load arguments
        raise RuntimeError("You have either to choose parameter -init or -load")

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


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-init", action="store_true")
    parser.add_argument("-load", action="store_true")
    parser.add_argument("-train", action="store_true")
    parser.add_argument("-sg_model_path", type=str, default=DEFAULT_SKIPGRAM_STORE_MODEL_PATH)
    parser.add_argument("-sg_model_name", type=str, default=DEFAULT_SKIPGRAM_MODEL_NAME)
    parser.add_argument("-save_kv", action="store_true")
    parser.add_argument("-epochs", type=int, default=1)
    parser.add_argument("-similarity", type=str, default="")
    parser.add_argument("-acc", action="store_true")

    args = parser.parse_args()

    # get model configuration
    stream = open(os.path.join(args.sg_model_path, args.sg_model_name, SKIPGRAM_MODEL_CONFIG_FILE), 'r')
    sg_model_config = yaml.load(stream, Loader=Loader)

    # execute skipgram
    skipgram(init=args.init,
             load=args.load,
             sg_model_path=args.sg_model_path,
             sg_model_name=args.sg_model_name,
             save_kv=args.save_kv,
             sg_model_config=sg_model_config,
             train=args.train,
             epochs=args.epochs,
             similarity=args.similarity,
             accuracy=args.acc)
