import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

myFeature = ([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0])
myLabel   = ([5.0, 8.8, 9.6, 14.2, 18.8, 19.5, 21.4, 26.8, 28.9, 32.0, 33.8, 38.2])

# plt.scatter(my_feature, my_label)

X = tf.constant(myFeature)
y = tf.constant(myLabel)
print(X)


my_model = tf.keras.Sequential([
    tf.keras.layers.Dense(input_shape=(1, ), units=1)
])

# my_model.compile(optimizer="Adam", loss="mse")
my_model.compile(optimizer=tf.keras.optimizers.RMSprop(lr=0.03), loss="mse")


history = my_model.fit(X, y, epochs=150, batch_size=10)

print(history.history)
print(my_model.get_weights())

plt.scatter(X, y)
plt.plot(X, my_model.predict(X))
plt.show()

