import tensorflow as tf
import numpy as np

fashion_mnist = tf.keras.datasets.fashion_mnist

train_data, test_data = fashion_mnist.load_data()

# train_images shape is (60000, 28,28)
train_images, train_labels = train_data

# test_images shape is (10000, 28,28)
test_images, test_labels = test_data

# we need to normalize our rgb values
train_images = train_images / 255
test_images = test_images / 255

# model
model = tf.keras.Sequential([
    tf.keras.layers.Flatten(input_shape=(28,28)),
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dense(10, activation="softmax")
])

# model compile
model.compile(optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"])

# train
model.fit(train_images, train_labels, epochs=10)

# preds
preds = model.predict(test_images)
preds[0]
np.argmax(preds[0])
test_labels[0]

model.evaluate(test_images, test_labels)
