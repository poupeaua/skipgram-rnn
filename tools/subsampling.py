from preprocessing import load_reviews_npy, PROJECT_DIR,
import random
import numpy as np

file_path = PROJECT_DIR + 'data\\reviews_as_arrays\\test10.npy'

test = load_reviews_npy(file_name=file_path)

vocabulary_path = PROJECT_DIR + 'data\\aclImdb\\imdb.vocab'

SPECIAL_CHARS = "!@#$%^&*()[]{};:,./<>?|`~-=_+\n\t\\"


def parse_vocabulary_as_dict(v_path):
    vocabulary = dict()
    with open(v_path, 'r', encoding='utf-8', ) as words:
        for w in words:
            w = w.translate({ord(c): "" for c in SPECIAL_CHARS})
            if w != '':
                vocabulary[w] = 0
    return vocabulary


def count_words_in_review(vocabulary_dict):
    :
        if word in vocabulary_dict:
            vocabulary_dict[word] += 1
        # else:
        #     print(word)


def discount_prob(word_str, v_dict, v_len, t=1e-5):
    q = 0
    if word_str in v_dict:
        if v_dict[word_str] > 0:
            f_w = v_dict[word_str] / v_len
            q = (t / f_w) ** 0.5
    return 1 - q  # p


def random_dropout(prob):
    return random.random() <= prob


def subsample(rev, v, v_length):
    new_rev = []
    for w in rev:
        dp = discount_prob(w, v, v_length)
        if not random_dropout(dp):
            new_rev.append(w)
    return np.array(new_rev, dtype="U")


if __name__ == '__main__':
    my_vocabulary = parse_vocabulary_as_dict(v_path=vocabulary_path)
    for review in test:
        count_words_in_review(review, my_vocabulary)
    length = len(my_vocabulary)
    print(test[0])
    print("-----------------\n")
    print(subsample(test[0], my_vocabulary, length))
    # for i, w in enumerate(test[0]):
    #     dp = discount_prob(w, my_vocabulary, length)
    #     print(i, w)
    #     print(dp)
    #     print(random_dropout(dp))
