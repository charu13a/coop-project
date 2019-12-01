#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Mar 25 14:05:19 2019

@author: charuagarwal
"""

import cv2
import tensorflow as tf
from tensorflow.keras.models import Model
import matplotlib.pyplot as plt
import numpy as np
import os

os.environ['KMP_DUPLICATE_LIB_OK']='True'

s = 974183652651274389283596714129835476746912538835647921568329147317468295492751863
sudoku = [int(i)/10 for i in str(s)]

sudoku = np.array([sudoku])
sudoku = sudoku.reshape(1, 9, 9, 1)

model = tf.keras.models.load_model("best_model.h5")
prediction = model.predict([sudoku]*3)
print(prediction)

intermediate_layer_model = Model(inputs=model.input,
                                 outputs=model.get_layer('gaussian_noise_1').output)
intermediate_output = intermediate_layer_model.predict([sudoku]*3)

print(intermediate_output)
