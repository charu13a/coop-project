#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Mar 25 14:05:19 2019

@author: charuagarwal
"""

import cv2
import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
import os

def convertToInt(word):
	new_word=[]
	for ch in word:
		new_word.append((ord(ch)-65)/26)
	return new_word

def convertToChar(word):
	new_word=[]
	for ch in word:
		new_word.append((chr(int(round(((ch*26 + 65)))))))
	return new_word

os.environ['KMP_DUPLICATE_LIB_OK']='True'

words = ['S', 'H', 'O', 'E', ' ', 'M', 'A', 'R', 'X', ' ', 'U', 'R', 'G', 'E', ' ', 'T', 'K', 'O', 'S', ' ', 'S', 'M', 'U', 'T', ' ', 'H', 'A', 'R', 'K',' ' , 'O', 'R', 'G', 'O', ' ', 'E', 'X', 'E', 'S', ' ']
processed_words = convertToInt(words)

model = tf.keras.models.load_model("crossword_ffnn.model")
prediction = model.predict(np.array([processed_words]))
print(convertToChar(list(prediction[0])))