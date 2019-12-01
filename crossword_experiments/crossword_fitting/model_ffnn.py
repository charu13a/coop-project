import numpy as np
from sklearn.model_selection import train_test_split
from keras.models import Sequential
from keras.layers import Dense

# Feed Forward Neural Network to generate crossword from a given set of words.

# Converts a list of characters to integer representation
def convertToInt(word):
	new_word=[]
	for ch in word:
		new_word.append((ord(ch)-65)/26)
	return new_word

X = []
Y = []

for i in range(100000):
	# read words
	with open('gen_words/word'+str(i)+'.txt', 'r') as f:
	    words = f.read().split('\n')
	    f.close()
	words.pop()
	# read grid
	with open('gen_crosswords/crossword'+str(i)+'.txt', 'r') as f:
	    grid = f.read().split(' ')
	    f.close()
	grid.pop()
	# process the input and output
	processed_words = list(map(lambda word: convertToInt(list(word)), words))
	words_flat_list = [item for sublist in processed_words for item in sublist]
	processed_grid = convertToInt(grid)
	X.append(words_flat_list)
	Y.append(processed_grid)

X = np.array(X)
Y = np.array(Y)

X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.1)

model = Sequential()  #import Sequential NN model 

model.add(Dense(32, input_dim=len(X[0]), activation='relu'))  # 1st hidden layer
model.add(Dense(len(Y[0]), activation='softmax'))  # output layer
model.compile(loss='categorical_crossentropy', optimizer='sgd', metrics=['accuracy'])  # compile the model
model_log=model.fit(X_train, y_train, epochs=10, batch_size=200, validation_split=0.10) 

score = model.evaluate(X_test, y_test, verbose=1)

print('Test loss:', score[0])
print('Test accuracy:', score[1])

model.summary()
model.save("crossword_ffnn.model")  