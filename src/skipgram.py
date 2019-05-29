from gensim.test.utils import common_texts, get_tmpfile
from gensim.models import Word2Vec
import multiprocessing
import logging
import os


# API :
# https://radimrehurek.com/gensim/models/base_any2vec.html#gensim.models.base_any2vec.BaseWordEmbeddingsModel
# https://radimrehurek.com/gensim/models/word2vec.html
# interesting codes :
# https://github.com/kavgan/nlp-in-practice/blob/master/word2vec/Word2Vec.ipynb

cpu_count = multiprocessing.cpu_count()

def main():

    # sentences / corpus = None so the model is left uninitialized
    model = Word2Vec(None,
                     size=size,
                     window=window,
                     min_count=min_count,
                     hs=hs,
                     negative=negative)

    # train
    model.train(sentences=sentences,
                total_examples=None,
                total_words=None,
                epochs=None)

    # save
    model.save(os.path.join(save_path, save_name))

    # load
    model.load(os.path.join(save_path, save_name))

    # save vectors representation of words
    model.wv.save(path)

    # load vectors representation of words
    wv = KeyedVectors.load("model.wv", mmap='r')
