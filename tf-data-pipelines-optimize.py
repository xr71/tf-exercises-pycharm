import tensorflow as tf
import pandas as pd
import numpy as np
import os
import time
import datetime
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split


CSV_URL = "https://storage.googleapis.com/applied-dl/heart.csv"

file_path = tf.keras.utils.get_file("heart.csv", CSV_URL)
print(file_path)

df =pd.read_csv(file_path)
print(df.head())

#sns.pairplot(data=df, hue="target")
#plt.show()

df.pop("thal")

target = df.pop("target")
features = df.copy()

X_train,X_test,Y_train,Y_test = train_test_split(features, target, test_size=0.3, random_state=1)

def normalize(x, train_stats):
    return (x-train_stats["mean"]) / train_stats["std"]

train_stats = X_train.describe().transpose()

X_train_normed = normalize(X_train, train_stats)
X_test_normed = normalize(X_test, train_stats)

model = tf.keras.Sequential([
    tf.keras.layers.Dense(32, activation='relu', input_shape=(12,)),
    tf.keras.layers.Dense(8, activation='relu'),
    tf.keras.layers.Dense(1, activation='sigmoid'),
])

model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])

# history = model.fit(X_train_normed, Y_train, epochs=50)
# model.evaluate(X_test_normed, Y_test)


# now using tf data pipeline with batching
dataset = tf.data.Dataset.from_tensor_slices((X_train_normed.values, Y_train.values)).batch(64)

# model.fit(dataset, epochs=10)
# print(model.evaluate(X_test_normed, Y_test))


# same thing but now with prefetching 
dataset = tf.data.Dataset.from_tensor_slices((X_train_normed.values, Y_train.values)).batch(64).prefetch(2)

# model.fit(dataset, epochs=50)
# print(model.evaluate(X_test_normed, Y_test))



# parallel loading
# interleave

# dataset = tf.data.Dataset.interleave(dataset, num_parallel_calls=tf.data.experimental.AUTOTUNE)
