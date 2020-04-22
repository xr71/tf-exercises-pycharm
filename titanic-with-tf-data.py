import tensorflow as tf
import pandas as pd
import csv
import requests

################################################################################
# PART 1: loading tf data from csv
################################################################################
TRAIN_DATA_URL = "https://storage.googleapis.com/tf-datasets/titanic/train.csv"
TEST_DATA_URL = "https://storage.googleapis.com/tf-datasets/titanic/eval.csv"

# use keras utils to get the files
train_file_path = tf.keras.utils.get_file("train.csv", TRAIN_DATA_URL)
test_file_path = tf.keras.utils.get_file("eval.csv", TEST_DATA_URL)

LABEL_COLUMN = 'survived'
LABELS = [0, 1]

# helper function for makign csv dataset object
def get_dataset(file_path, **kwargs):
    dataset = tf.data.experimental.make_csv_dataset(
        file_path,
        batch_size=5,
        label_name=LABEL_COLUMN,
        na_value="?",
        num_epochs=1,
        ignore_errors=True,
        **kwargs
    )

    return dataset


# this creates PrefetchDataset objects - makes column names to datatypes
raw_train_data = get_dataset(train_file_path)
raw_test_data = get_dataset(test_file_path)

# to take a batch out
for batch, labels in raw_train_data.take(1):
    print("################################################################################", "features batch")
    print(batch)
    print()
    print("################################################################################", "labels")
    print(labels)



################################################################################
# use this if your csv file is not column labeled
CSV_COLUMNS = ['survived', 'sex', 'age', 'n_siblings_spouses', 'parch', 'fare', 'class', 'deck', 'embark_town', 'alone']

temp_dataset = get_dataset(train_file_path, column_names=CSV_COLUMNS)

################################################################################



################################################################################
# PART 2: loading tf data from pandas
################################################################################
import pandas as pd
df = pd.read_csv(TRAIN_DATA_URL)
df.head()

selected_cols = ["survived", "sex", "age", "fare", "class", "deck", "alone"]
df = df[selected_cols]
df["sex"] = pd.Categorical(df["sex"]).codes
df["class"] = pd.Categorical(df["class"]).codes
df["deck"] = pd.Categorical(df["deck"]).codes
df["alone"] = pd.Categorical(df["alone"]).codes

target = df.pop("survived")
features = df.copy()


# create tensorflow dataset
dataset = tf.data.Dataset.from_tensor_slices((features.values, target.values))

# dataset is now a TensorSliceDataset object




################################################################################
# PART 3: TFExample
################################################################################
# we can continue using the numeric pandas df from above
feat1 = features["sex"]
feat2 = features["age"]
feat3 = features["class"]
feat4 = features["alone"]


feature_ds = tf.data.Dataset.from_tensor_slices((feat1, feat2, feat3, feat4))
for f0,f1,f2,f3 in feature_ds.take(1):
    print(f0,f1,f2,f3)





################################################################################
# PART 4: TFRecord
################################################################################

# map the serialize function over the entire dataset



################################################################################
# PART 5: prepping data and working with Image data
################################################################################
import seaborn as sns
from sklearn.model_selection import train_test_split

df = pd.read_csv("banknotes.csv")
sns.pairplot(df, hue="class")

target = df.pop("class")
features = df.copy()

dataset = tf.data.Dataset.from_tensor_slices((features.values, target.values))
train_dataset = dataset.shuffle(len(df)).batch(32)

model = tf.keras.Sequential([
    tf.keras.layers.Dense(8, activation="relu"),
    tf.keras.layers.Dense(1, activation="sigmoid")
])

model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])

model.fit(train_dataset, epochs=100)