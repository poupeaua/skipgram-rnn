from sklearn.datasets import load_files
import gensim
import numpy as np
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

env = yaml.load(open(os.path.join(skipgram_rnn_path, "env.yml"), 'r'), Loader=Loader)
# ---------------------------------------------------------------------------

NB_REVIEWS = 50000

PROJECT_PATH = env["project_abspath"]

PATHS_DATA = [os.path.join(PROJECT_PATH, "data/aclImdb/test/pos"),
              os.path.join(PROJECT_PATH, "data/aclImdb/test/neg"),
              os.path.join(PROJECT_PATH, "data/aclImdb/train/pos"),
              os.path.join(PROJECT_PATH, "data/aclImdb/train/neg")]
PATH_STORE_REVIEWS_AS_ARRAYS = os.path.join(PROJECT_PATH, "data/reviews_as_arrays/")
DEFAULT_STORE_FILENAME = "test.npy"  # reviews_as_arrays.npy


# ----------------------------------------------------------------------------


def sentence_preprocess():
    # TODO:
    pass


def iter_reviews_file(paths=PATHS_DATA):
    """
        Iterate over all the paths to reviews file.

        Arguments:
            paths (list) : list of paths to find all the reviews .txt files.

        Returns:
            yield (str) : iteration over all the absolute paths of the files
    """
    for path in paths:
        for file in os.listdir(path):
            yield os.path.join(path, file)


def read_reviews(nb_files, disp_info_iter=1000):
    """
        This method reads the .txt files in path.

        Arguments:
            nb_files (int) : number of review files to read.
            disp_info_iter (int) : display information every disp info iter.

        Returns:
            stock (array) : array that contains lists of words.
                One list for each review.
    """
    logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)

    logging.info("reading reviews...this may take a while")

    stock = list()
    for i, filepath in enumerate(iter_reviews_file()):
        if (i % disp_info_iter) == 0:
            logging.info("read {0} reviews".format(i))
        with open(file=filepath) as f:
            for line in f:
                # do some pre-processing and return a list of words for each review text
                stock.append(np.array(gensim.utils.simple_preprocess(line), dtype="U"))
        if i >= nb_files - 1:
            break

    logging.info("Done reading data file")

    return np.array(stock)


def save_reviews_npy(nb_files=NB_REVIEWS,
                     path=PATH_STORE_REVIEWS_AS_ARRAYS,
                     file_name=DEFAULT_STORE_FILENAME):
    """
        Does not return anything but save the array in a file.

        Arguments:
            nb_files (int) : number of reviews to get and to save
            path (str) : path to save the array file .npy.
            file_name (str) : name of the file
    """
    # read the tokenized reviews into a list
    # each review item becomes a series of words
    # so this becomes a list of lists
    reviews_as_arrays = read_reviews(nb_files=nb_files)
    np.save(file=os.path.join(path, file_name), arr=reviews_as_arrays)


def load_reviews_npy(path=PATH_STORE_REVIEWS_AS_ARRAYS,
                     file_name=DEFAULT_STORE_FILENAME):
    """
        Arguments
            path (str) : path to look for the array file .npy.
            file_name (str) : name of the file to look for in the path.
    """
    return np.load(file=os.path.join(path, file_name), allow_pickle=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-nf", type=int, required=True)
    parser.add_argument("-file_name", type=str, required=True)

    args = parser.parse_args()

    save_reviews_npy(nb_files=args.nf, file_name=args.file_name)
    test = load_reviews_npy(file_name=args.file_name)
    print(test)
