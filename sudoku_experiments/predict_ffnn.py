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

os.environ['KMP_DUPLICATE_LIB_OK']='True'

s = 289765421317924856645138729763891542521473968894652173432519687956387214178246395
sudoku = [int(i)/10 for i in str(s)]

model = tf.keras.models.load_model("sudoku_ffnn.model")
prediction = model.predict(np.array([sudoku]))
print(prediction)