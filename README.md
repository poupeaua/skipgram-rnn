# skipgram-rnn
Study of the Skipgram model for word embeddings. Then evaluating different distributed representations of the words on a film review project using a many-to-one dynamic RNN.

- Skipgram : Word2Vec doc https://radimrehurek.com/gensim/models/word2vec.html
- Dynamic RNN : Keras doc https://keras.io/layers/recurrent/

## Environnement
1 - Download data here http://ai.stanford.edu/~amaas/data/sentiment/ (turn into a directory named *aclImdb/* after decompression).
Create a directory named *data/* at the project root. Put the directory *aclImdb/* in *data/*.

Structure:
```
  -acllmdb
    |-test
        |- neg 12500 files
        |- pos 12500 files
    |-train
        |- neg 12500 files
        |- pos 12500 files
        |- unsup 50000 files - unlabeled data
```

 In this labeled train/test set, a negative review has a score <= 4 out of 10, and a positive review has a score >= 7 out of 10. Thus reviews with more neutral ratings are not included in the train/test sets. We labeled a negative review whose score <= 4 as 0, and a positive review whose score >= 7 as 1.

2 - Create a env.yml file at the project root and write in it : project_abspath: "absolute/path/to/skipgram-rnn"

## How to Skipgram ?
3 - You are ready to train your first Skipgram model ! Create a directory *my_skipgram_model/* in *models/skipgram*. Create a config.yml file in *my_skipgram_model/* and indicate in it the following parameters : size, window, hs, negative, sample, min_count: 1, workers. 

4 - **python3 src/skipgram.py -init -sg_model_name my_skipgram_model** initialize the model and then train it with one epoch. 

5 - **python3 src/skipgram.py -load -sg_model_name my_skipgram_model -train -epoch 1** load your pre-trained model and train it 1 time on the whole dataset. Saving is automatic after initialization and training.

6 - **python3 src/skipgram.py -load -sg_model_name my_skipgram_model -similarity my_word** display the 10 words that are the closest to my_word (recommendation use my_word: good, bad, awesome etc...). It is just to test the model empirically.

7 - **python3 src/skipgram.py -load -sg_model_name my_skipgram_model -save_kv** store your keyed vectors or word embeddings in a *models/skipgram/my_skipgram_model/my_skipgram_model.kv*.

## How to Dynamic RNN ?

8 - You are ready to train your first RNN model ! Create a directory *my_rnn_model/* in *models/rnn*.

9 - **python3 src/rnn.py -init -rnn_model_name my_rnn_model -sg_model_name my_skipgram_model** initialize the model. 

10 - **python3 src/rnn.py -load -rnn_model_name my_rnn_model -sg_model_name my_skipgram_model -train -training_size 7500 -epochs 1** use the word embeddings of my_skipgram_model in order to get the vector representation of each word in all reviews and train the RNN 1 time on 7500 (max 25000) random training reviews.

11 - **python3 src/rnn.py -load -rnn_model_name my_rnn_model -sg_model_name my_skipgram_model -test -testing_size 5000** test the model efficiency on 5000 (max 25000) random testing reviews. Display the confusion matrix and the accuracy of the current model.

12 - **python3 src/rnn.py -load -rnn_model_name my_rnn_model -sg_model_name my_skipgram_model -pred** take a random test review, display it and print its ground truth class, the predicted class and the probability of being a good review.

Have fun !
