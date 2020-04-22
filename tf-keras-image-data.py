import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import os

IMAGE_URL = "https://storage.googleapis.com/mledu-datasets/cats_and_dogs_filtered.zip"

img_path = tf.keras.utils.get_file('cats_and_dogs.zip', IMAGE_URL, extract=True)

train_img_files = os.path.join('/home/xuren/.keras/datasets', 'cats_and_dogs_filtered', 'train')

BATCH = 128
IMG_HEIGHT = 150
IMG_WIDTH = 150

train_image_gen = ImageDataGenerator(rescale=1.0/255.0)
train_data_gen = train_image_gen.flow_from_directory(batch_size=BATCH,
                                            directory=train_img_files,
                                            shuffle=True,
                                            target_size=(IMG_HEIGHT, IMG_WIDTH))


def plot_image(img):
    plt.figure()
    plt.imshow(img)
    plt.colorbar()
    plt.grid(False)
    plt.show()


#for i in range(5):
#    _img = train_data_gen[i][0][0]
#    plot_image(_img)


# data augmentation
# methods
## rotation range
## width shift range
## height shift range
## horizontal flip
## zoom range

train_image_gen_2 = ImageDataGenerator(rescale=1.0/255.0,
                                      rotation_range=45,
                                      width_shift_range=0.15,
                                      height_shift_range=0.15,
                                      horizontal_flip=True)

train_data_gen_2 = train_image_gen_2.flow_from_directory(batch_size=BATCH,
                                            directory=train_img_files,
                                            shuffle=True,
                                            target_size=(IMG_HEIGHT, IMG_WIDTH))
for i in range(5):
    _img = train_data_gen_2[0][0][0]
    plot_image(_img)

