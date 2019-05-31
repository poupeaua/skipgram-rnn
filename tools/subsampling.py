from preprocessing import loadReviewsNpy, PROJECT_DIR

file_path = PROJECT_DIR + 'data\\reviews_as_arrays\\test10.npy'

test = loadReviewsNpy(file_name=file_path)

vocab_path = PROJECT_DIR + 'data\\aclImdb\\imdb.vocab'


def parse_vocab_as_dict(vocab_path):
    vocab = dict()
    with open(vocab_path, 'r', encoding='utf-8', ) as words:
        for w in words:
            vocab[w.replace('\n', '')] = 0
    return vocab


if __name__ == '__main__':
    vocab = parse_vocab_as_dict(vocab_path=vocab_path)
    print(vocab)
