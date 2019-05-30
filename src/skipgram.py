from gensim.test.utils import common_texts, get_tmpfile
from gensim.models import Word2Vec
import multiprocessing
import logging
import os
import argparse
import yaml


# API :
# https://radimrehurek.com/gensim/models/base_any2vec.html#gensim.models.base_any2vec.BaseWordEmbeddingsModel
# https://radimrehurek.com/gensim/models/word2vec.html
# interesting codes :
# https://github.com/kavgan/nlp-in-practice/blob/master/word2vec/Word2Vec.ipynb

cpu_count = multiprocessing.cpu_count()


# ----------------------------------------------------------------------------

DEFAULT_STORE_MODEL_PATH = "/u/a/2019/apoupeau/Documentos/recpatr/skipgram-rnn/models"
DEFAULT_MODEL_NAME = "model_test"

#  ----------------------------------------------------------------------------


def skipgram(load,
             model_path,
             model_name,
             save_embeddings):

    # sentences / corpus = None so the model is left uninitialized
    model = Word2Vec(None,
                     size=size,
                     window=window,
                     min_count=min_count,
                     hs=hs,
                     negative=negative)

    # save
    model.save(os.path.join(save_path, save_name))

    # train
    model.train(sentences=sentences,
                total_examples=None,
                total_words=None,
                epochs=None)

    # load
    model.load(os.path.join(save_path, save_name))

    # save vectors representation of words
    model.wv.save(path)

    # load vectors representation of words
    wv = KeyedVectors.load("model.wv", mmap='r')



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-load", action="store_true")
    parser.add_argument("-model_path", type=str, required=True, default=DEFAULT_STORE_MODEL_PATH)
    parser.add_argument("-model_name", type=str, required=True, default=DEFAULT_MODEL_NAME)
    parser.add_argument("-save_embeddings", action="store_true")

    args = parser.parse_args()

    model_config = yaml.dump(yaml.load(os.path.join(model_path, model_name)))

    skipgram(load=args.load,
             model_path=args.model_path,
             model_name=args.model_name,
             save_embeddings=args.save_embeddings)
