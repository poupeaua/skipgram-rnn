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
PATHS_TRAIN_DATA = [os.path.join(PROJECT_PATH, "data/aclImdb/train/pos"),
                      os.path.join(PROJECT_PATH, "data/aclImdb/train/neg")]
PATHS_TEST_DATA = [os.path.join(PROJECT_PATH, "data/aclImdb/test/pos"),
                      os.path.join(PROJECT_PATH, "data/aclImdb/test/neg")]
PATHS_DATA = PATHS_TRAIN_DATA + PATHS_TEST_DATA
PATH_STORE_REVIEWS_AS_ARRAYS = os.path.join(PROJECT_PATH, "data/reviews_as_arrays/")
DEFAULT_STORE_FILENAME = "test.npy"  # reviews_as_arrays.npy


# ----------------------------------------------------------------------------


def sentence_preprocess():
    ## TODO:
    return


def iter_reviews_file(paths=PATHS_DATA):
    """
        Iterate over all the paths to reviews file.

        Arguments:
            paths (list) : list of paths to find all the reviews .txt files.

        Return:
            yield (str) : iteration over all the absolute paths of the files
    """
    for path in paths:
        for file in os.listdir(path):
            yield os.path.join(path, file)


def iter_random_reviews_file(paths):
    """
        Iterate randomly over all the paths to reviews file.
        USE THIS FUNCTION TO TRAIN THE RNN MODEL IN UNIFORM WAY.
        (not only positive or only negative output).

        Arguments:
            paths (list) : list of paths to find all the reviews .txt files.

        Return:
            yield (str) : iteration over all the absolute paths of the files
    """
    review_paths = list(iter_reviews_file(paths=paths))
    np.random.shuffle(review_paths)
    for review_path in review_paths:
        yield review_path


def get_inout_from_review(review_path, words_embeddings, good_reviews_min_value=7):
    """
        Get the input and output for the rnn dynamic model given a reivew path

        Arguments:
            review_path (str) : path to review
            words_embeddings (dict) : KeyedVectors object.
                It allows to associate a vector to a word.
            good_reviews_min_value (int) : threshold between good and bad review

        Return:
            input (array) : numpy array. It has the shape (N, S) where N is the
                number of word in the review and S the size of the vector
                representation of each word.
            label (int) : 0 or 1 for a respective bad or good film review
    """
    if review_path[-6] == "1":
        # case of 10
        label = 1
    elif review_path[-6] != "1" and int(review_path[-5]) >= good_reviews_min_value:
        # case of number between 0 and 9
        label = 1
    else:
        label = 0
    with open(file=review_path) as f:
        # do some pre-processing and return a list of words for each review text
        tokenized_review = gensim.utils.simple_preprocess(f.read())
    for j, word in enumerate(tokenized_review):
        tokenized_review[j] = words_embeddings[word]
    input = np.array(tokenized_review)
    return input, label


def iter_reviews_as_model_inout(words_embeddings,
                                paths,
                                max_nb_reviews,
                                good_reviews_min_value=7):
    """
        Transform a review into a rnn trainable object (input, label).

        Arguments:
            words_embeddings (dict) : KeyedVectors object.
                It allows to associate a vector to a word.
            paths (str) : paths to reviews to consider.
            max_nb_reviews (int) : maximum number of review to iterate.
            good_reviews_min_value (int) : threshold between good and bad review

        Yield:
            input (array) : numpy array. It has the shape (N, S) where N is the
                number of word in the review and S the size of the vector
                representation of each word.
            label (int) : 0 or 1 for a respective bad or good film review
    """
    for i, review_path in enumerate(iter_random_reviews_file(paths=paths)):
        yield get_inout_from_review(review_path=review_path,
                                    words_embeddings=words_embeddings)

        # stop yield when the number of iterations is bigger than max_nb_reviews
        if i >= max_nb_reviews - 1:
            break


def read_reviews(nb_files, disp_info_iter=1000):
    """
        This method reads the .txt files in path.

        Arguments:
            nb_files (int) : number of review files to read.
            disp_info_iter (int) : display information every disp info iter.

        Return:
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

# txt = open(file="data/aclImdb/test/pos/6326_8.txt").read()
# print(txt)
