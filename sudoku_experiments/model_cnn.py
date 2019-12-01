# Convolutional Neural Network to classify a sudoku as valid or invalid.

# importing modules
import csv
from keras.models import Sequential
from keras.layers import Dense, Flatten
from keras.layers import Conv2D
from keras.utils import to_categorical
import matplotlib.pyplot as plt
import numpy as np
import os
import random
from random import randrange
from sklearn.model_selection import train_test_split

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
    for i in range(randrange(81)):
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

model = Sequential()  #import Sequential NN model 
model.add(Conv2D(9, kernel_size=(3, 3),
                 activation='relu',
                 input_shape=input_shape, strides=3))
model.add(Flatten())
model.add(Dense(32, activation='relu'))
model.add(Dense(len(Y[0]), activation='softmax'))
model.compile(loss='categorical_crossentropy', optimizer='sgd', metrics=['accuracy']) # compile the model
model_log=model.fit(X_train, y_train, epochs=5, batch_size=200, validation_split=0.10) 

score = model.evaluate(X_test, y_test, verbose=1)

print('Test loss:', score[0])
print('Test accuracy:', score[1])

model.summary() 
model.save("sudoku_cnn.model")  

layers = model.layers
#plot the filters
fig,ax = plt.subplots(nrows=1,ncols=2)
for i in range(1):
    ax[i].imshow(layers[i].get_weights()[0][:,:,:,0][:,:,0],cmap='gray')
    ax[i].set_title('block'+str(i+1))
    ax[i].set_xticks([])
    ax[i].set_yticks([])