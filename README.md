# skipgram-rnn
Study of the skipgram model for words embedding. Then evaluating different distributed representations of the words on a film review project using a many-to-one RNN.

# dataset
http://ai.stanford.edu/~amaas/data/sentiment/

Structure:
```
  -acllmdb
    |-test
        |- neg 12500 files
        |- pos 12500 files
    |-train
        |- neg 12500 files
        |- pos 12500 files
        |- unsup 50000 files
```

 In this labeled train/test sets, a negative review has a score <= 4 out of 10, and a positive review has a score >= 7 out of 10. Thus reviews with more neutral ratings are not included in the train/test sets. We labeled a negative review whose score <= 4 as 0, and a positive review whose score >= 7 as 1.
