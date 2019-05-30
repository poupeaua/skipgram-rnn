from sklearn.datasets import load_files
import gensim
import numpy as np
import logging
import os
import argparse

logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)

# ---------------------------------------------------------------------------

NB_REVIEWS = 50000

PATHS_DATA = ["/u/a/2019/apoupeau/Documentos/recpatr/skipgram-rnn/data/aclImdb/test/pos",
         "/u/a/2019/apoupeau/Documentos/recpatr/skipgram-rnn/data/aclImdb/test/neg",
         "/u/a/2019/apoupeau/Documentos/recpatr/skipgram-rnn/data/aclImdb/train/pos",
         "/u/a/2019/apoupeau/Documentos/recpatr/skipgram-rnn/data/aclImdb/train/neg"]

PATH_STORE_REVIEWS_AS_ARRAYS = "/u/a/2019/apoupeau/Documentos/recpatr/skipgram-rnn/data/reviews_as_arrays/"

DEFAULT_STORE_FILENAME = "reviews_as_arrays.npy"

# ----------------------------------------------------------------------------



def sentence_preprocess():
    ## TODO:
    return



def iterReviewsFile(paths=PATHS_DATA):
    """
        Arguments:
            paths (list) : list of paths to find all the reviews .txt files.

        Returns:
            yield (str) : iteration over all the absolute paths of the files
    """
    for path in paths:
        for file in os.listdir(path):
            yield os.path.join(path, file)



def readReviews(nb_files, disp_info_iter=1000):
    """
        This method reads the .txt files in path.

        Arguments:
            nb_files (int) : number of review files to read.
            disp_info_iter (int) :

        Returns:
            stock (array) : array that contains lists of words.
                One list for each review.
    """

    logging.info("reading reviews...this may take a while")

    stock = list()
    for i, filepath in enumerate(iterReviewsFile()):
        if (i%disp_info_iter) == 0:
            logging.info("read {0} reviews".format(i))
        with open(file=filepath) as f:
            for line in f:
                # do some pre-processing and return a list of words for each review text
                 stock.append(np.array(gensim.utils.simple_preprocess(line), dtype="U"))
        if i >= nb_files-1:
            break

    logging.info("Done reading data file")

    return np.array(stock)



def saveReviewsNpy(nb_files=NB_REVIEWS,
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
    reviews_as_arrays = readReviews(nb_files=nb_files)
    np.save(file=os.path.join(path, file_name), arr=reviews_as_arrays)



def loadReviewsNpy(path=PATH_STORE_REVIEWS_AS_ARRAYS,
                   file_name=DEFAULT_STORE_FILENAME):
    """
        Arguments
            path (str) : path to look for the array file .npy.
            file_name (str) : name of the file to look for in the path.
    """
    return np.load(file=os.path.join(path, file_name))



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-nf", type=int, required=True)
    parser.add_argument("-file_name", type=str, required=True)

    args = parser.parse_args()

    saveReviewsNpy(nb_files=args.nf, file_name=args.file_name)
    test = loadReviewsNpy(file_name=args.file_name)


# txt = open(file="data/aclImdb/test/pos/6326_8.txt").read()
# print(txt)

# reviews_train = load_files("aclImdb/train/")
# text_train, y_train = reviews_train.data, reviews_train.target
#
# print("Number of documents in train data: {}".format(len(text_train)))
# print("Samples per class (train): {}".format(np.bincount(y_train)))

# reviews_test = load_files("aclImdb/test/")
# text_test, y_test = reviews_test.data, reviews_test.target
#
# print("Number of documents in test data: {}".format(len(text_test)))
# print("Samples per class (test): {}".format(np.bincount(y_test)))
