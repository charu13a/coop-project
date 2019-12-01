# Experiments for crosswords
Two major problems:  
crossword_fitting/ : to test if a neural network can fit a crossword from a given set of words.  
crossword_suggestions/ : to suggest thematic words for crosswords.  

## crossword_fitting/
```
└── crossword_fitting
    ├── crossword_ffnn.model
    ├── gen_crosswords 
    ├── gen_random.cpp # run this to generate synthetic crosswords (4x4)
    ├── gen_words
    ├── model_ffnn.py # train model on words and crosswords
    └── predict_ffnn.py
```
## crossword_suggestions/
```
└── crossword_suggestions
    ├── CoOp_crossword2vec.ipynb # end-to-end algorithm for recommending words
    ├── CoOp_crossword2vec_retrained.ipynb # training on custom dataset with weights initialized with Google News
    ├── CoOp_word2vec_custom.ipynb # training on custom dataset
    └── CoOp_word2vec_examples.ipynb # using word2vec to give suggestions for different domain crosswords
```

Note: To view any .ipynb file (python notebook), use jupyter notebook <filename.ipynb>.