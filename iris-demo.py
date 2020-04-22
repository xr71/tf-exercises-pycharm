import csv
import tensorflow as tf
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import pytest
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelBinarizer, normalize

df = pd.read_csv(
    "https://gist.githubusercontent.com/netj/8836201/raw/6f9306ad21398ea43cba4f7d537619d0e07d5ae3/iris.csv")

data = df[["petal.length", "petal.width", "sepal.length", "sepal.width"]]
data = normalize(data)
lb = LabelBinarizer()
label = lb.fit_transform(df[["variety"]])

X_train, X_test, y_train, y_test = train_test_split(data, label, test_size=0.4)


# sns.scatterplot(x="petal.length", y="petal.width", hue="variety", data=df)

def test_csv_len():
    assert len(df) == 150


print(X_train.shape)
print(y_train.shape)
print(X_test.shape)
print(y_test.shape)

model = tf.keras.Sequential([
    tf.keras.layers.Dense(32, input_shape=(4,), activation="relu"),
    tf.keras.layers.Dense(8, input_dim=32, activation="relu"),
    tf.keras.layers.Dense(3, activation="softmax")
])

model.compile(optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"])

history = model.fit(X_train, y_train, epochs=150, batch_size=32)

model.predict(X_test)

model.evaluate(X_test, y_test)
