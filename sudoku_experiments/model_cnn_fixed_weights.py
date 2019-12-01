# Convolutional Neural Network to classify a sudoku as valid or invalid.
# The weights are fixed for now.

# importing modules
import csv
from keras.models import Sequential, Model
from keras.layers import Dense, Flatten, Conv2D, Concatenate, GaussianNoise
from keras.utils import to_categorical
import matplotlib.pyplot as plt
import numpy as np
import os
import random
from random import randrange
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score
from sklearn.metrics import cohen_kappa_score
from sklearn.metrics import roc_auc_score
from sklearn.metrics import confusion_matrix

# csv file name 
filename = "sudoku.csv"

sudoku = []
  
# reading csv file 
with open(filename, 'r') as csvfile: 
    data = csv.reader(csvfile)
    for row in data:
        sudoku.append(row[1])

sudoku = list(map(lambda x: [int(i)/10 for i in str(x)], sudoku))

# split the sudokus into valid and invalid
index = int(len(sudoku)/2)
valid = sudoku[0:index]
invalid = sudoku[index:]

def makeInvalid(s):
    for i in range(1):
        random_index = randrange(len(s)-1)
        diff = (randrange(8) + 1) / 10
        s[random_index] = (s[random_index]+diff)%1
    return s

# make invalid sudoku
invalid = list(map(lambda s: makeInvalid(s), invalid))

# add labels to sudoku
labelled_sudoku = []

for x in invalid:
    labelled_sudoku.append([x, 0])
for x in valid:
    labelled_sudoku.append([x, 1])

random.shuffle(labelled_sudoku)

os.environ['KMP_DUPLICATE_LIB_OK']='True'

# split labels and sudoku
X = []
Y = []

for x in labelled_sudoku:
    X.append(x[0])
    Y.append(x[1])

Y = to_categorical(Y, 2)

X = np.array(X)
Y = np.array(Y)

X = X.reshape(len(X), 9, 9, 1)
input_shape = (9, 9, 1)

X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.1)

# Conv2D to calculate each block sum
model_block_sum = Sequential()
model_block_sum.add(Conv2D(1, kernel_size=(3, 3),
                 activation='relu',
                 strides=3,
                 use_bias=False,
                 weights=[np.ones(shape=(3, 3, 1, 1))],
                 input_shape=input_shape))
model_block_sum.add(Flatten())
model_block_sum.layers[0].trainable = False

# Conv2D to calculate each column sum
model_col_sum = Sequential()
model_col_sum.add(Conv2D(1, kernel_size=(1, 9),
                 activation='relu',
                 use_bias=False,
                 weights=[np.ones(shape=(1, 9, 1, 1))],
                 input_shape=input_shape))
model_col_sum.add(Flatten())
model_col_sum.layers[0].trainable = False

# Conv2D to calculate each row sum
model_row_sum = Sequential()
model_row_sum.add(Conv2D(1, kernel_size=(9, 1),
                 activation='relu',
                 use_bias=False,
                 weights=[np.ones(shape=(9, 1, 1, 1))],
                 input_shape=input_shape))
model_row_sum.add(Flatten())
model_row_sum.layers[0].trainable = False

conc = Concatenate()([model_block_sum.output, model_row_sum.output, model_col_sum.output])
out = GaussianNoise(0.01)(conc)
out = Dense(32, activation='relu')(out)
out = Dense(32, activation='relu')(out)
out = Dense(len(Y[0]), activation='softmax')(out)

model = Model([model_block_sum.input, model_row_sum.input, model_col_sum.input], out)

model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy']) # compile the model
model_log=model.fit([X_train]*3, y_train, epochs=5, batch_size=200, validation_split=0.10) 

score = model.evaluate([X_test]*3, y_test, verbose=1)
print('Test loss:', score[0])
print('Test accuracy:', score[1])

# predict probabilities for test set
yhat_probs = model.predict([X_test]*3, verbose=1)
# predict crisp classes for test set
yhat_classes = np.argmax(yhat_probs, axis=1)

# reduce to 1d array
yhat_probs = [row[1] for row in yhat_probs]
yhat_test = [row[1] for row in y_test]

# accuracy: (tp + tn) / (p + n)
accuracy = accuracy_score(yhat_test, yhat_classes)
print('Accuracy: %f' % accuracy)
# precision tp / (tp + fp)
precision = precision_score(yhat_test, yhat_classes)
print('Precision: %f' % precision)
# recall: tp / (tp + fn)
recall = recall_score(yhat_test, yhat_classes)
print('Recall: %f' % recall)
# f1: 2 tp / (2 tp + fp + fn)
f1 = f1_score(yhat_test, yhat_classes)
print('F1 score: %f' % f1)

# kappa
kappa = cohen_kappa_score(yhat_test, yhat_classes)
print('Cohens kappa: %f' % kappa)
# ROC AUC
auc = roc_auc_score(yhat_test, yhat_probs)
print('ROC AUC: %f' % auc)
# confusion matrix
matrix = confusion_matrix(yhat_test, yhat_classes)
print(matrix)

model.summary() 
model.save("sudoku_cnn.model")  
