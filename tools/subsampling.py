import os
import sys
import random

import gensim
import numpy as np
import yaml

from preprocessing import iter_reviews_file

# allows import from skipgram-rnn directory
abspath_file = os.path.abspath(os.path.dirname(__file__))
skipgram_rnn_path = "/".join(abspath_file.split("/")[:-1])
sys.path.append(skipgram_rnn_path)

env = yaml.load(open(os.path.join(skipgram_rnn_path, "env.yml"), 'r'), Loader=yaml.Loader)

PROJECT_PATH = env["project_abspath"]

file_path = PROJECT_PATH + 'data/reviews_as_arrays/test10.npy'

vocabulary_path = PROJECT_PATH + 'data/aclImdb/imdb.vocab'

SPECIAL_CHARS = "!@#$%^&*()[]{};:,./<>?|`~-=_+\n\t\\"


def parse_vocabulary_as_dict(v_path):
    """
    Parse a .vocab txt file and converts it to a python dictionary

    Arguments:
        v_path (str) : the path of the .vocab file that contains the vocabulary

    Return:
        vocabulary (dict) : the dictionary that contains all the words in the vocab.
    """
    vocabulary = dict()
    with open(v_path, 'r', encoding='utf-8', ) as words:
        for w in words:
            w = w.translate({ord(c): "" for c in SPECIAL_CHARS})
            if w != '':
                vocabulary[w] = 0
    return vocabulary


def count_words(vocabulary_dict):
    """
    Update the frequency of every word in vocabulary_dict based on the imdb reviews, and returns the longest review.

    Arguments:
        vocabulary_dict (dict) : the dictionary that contains all the words in the vocab.

    Return:
        longest_review (str) : the longest review's path
        max_length (int) : the length of the longest review
    """
    words_counter = 0
    max_length = 0
    longest_review = ''
    for review_path in iter_reviews_file():
        with open(review_path, 'r', encoding='utf-8') as review:
            current_max_length = 0
            for line in review:
                l = gensim.utils.simple_preprocess(line)
                for word in l:
                    current_max_length += 1
                    words_counter += 1
                    if word in vocabulary_dict:
                        vocabulary_dict[word] += 1
                    else:
                        vocabulary_dict[word] = 0
            if current_max_length > max_length:
                max_length = current_max_length
                longest_review = review_path

    for w in vocabulary_dict:
        vocabulary_dict[w] /= words_counter

    return longest_review, max_length


def discount_prob(word, vocabulary, t=1e-5):
    """
    Compute the probability for the word to be discarded.

    Arguments:
        word (str) : the word which is going to be discarded.
        vocabulary (dict) : the dictionary containing the vocabulary.

    Return:
        p (float) : the discard word probability
    """

    q = 1
    if word in vocabulary:
        if vocabulary[word] > 0:
            f_w = vocabulary[word]
            q = (t / f_w) ** 0.5
    p = max(0, 1 - q)
    return p


def subsample(tokenized_review, vocabulary):
    j = len(tokenized_review) - 1
    while j >= 0:
        word = tokenized_review[j]
        if random.random() <= discount_prob(word, vocabulary):
            tokenized_review.pop(j)
        j -= 1
    return tokenized_review
