# Image grid Generator.
# --------------------------------------------------------------------------
#  About the code: Code use for generate the grid image of the image use for the paper
#  
# --------------------------------------------------------------------------
# @author : Yeuris Adolfo Lopez Jaime
# @date : October, 2023
# @Project : Predictive Maintanance using CCN for dectecting faults on Solar panels
# @Workplace : Universidad Central Del Este
# @City : San Pedro De Macoris, Republica Dominicana
# --------------------------------------------------------------------------

# Import all the library for use in the model

import pandas as pd 
import numpy as np 
import seaborn as sns
import matplotlib.pyplot as plt 

import tensorflow as tf
import random
from cv2 import resize
from glob import glob

# --------------------------------------------------------------------------


img_height = 244
img_width = 244
train_ds = tf.keras.utils.image_dataset_from_directory(
  r'/home/reloadzu/SolarPanels/Faulty_solar_panel',
  validation_split=0.2,
  subset='training',
  image_size=(img_height, img_width),
  batch_size=32,
  seed=42,
  shuffle=True)

val_ds = tf.keras.utils.image_dataset_from_directory(
  r'/home/reloadzu/SolarPanels/Faulty_solar_panel',
  validation_split=0.2,
  subset='validation',
  image_size=(img_height, img_width),
  batch_size=32,
  seed=42,
  shuffle=True)

class_names = train_ds.class_names
print(class_names)

plt.figure(figsize=(15, 15))
for images, labels in train_ds.take(2):
    for i in range(25):
        ax = plt.subplot(5, 5, i + 1)
        plt.imshow(images[i].numpy().astype("uint8"))
        plt.title(class_names[labels[i]])
        plt.axis("off")

plt.show()