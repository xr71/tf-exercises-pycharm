import tensorflow as tf
from tensorflow.estimator import LinearRegressor
import pandas as pd
from sklearn import datasets


data_load = datasets.load_boston()

feature_cols = data_load.feature_names
target_col = data_load.target
boston_data = pd.DataFrame(data_load.data, columns=feature_cols)

def make_input_fn(data_df, label_df, num_epochs=10, shuffle=True, batch_size=32):
  def input_function():
    ds = tf.data.Dataset.from_tensor_slices((dict(data_df), label_df))
    if shuffle:
      ds = ds.shuffle(1000)
    ds = ds.batch(batch_size).repeat(num_epochs)
    return ds
  return input_function

train_feed_input = make_input_fn(boston_data, target_col)


lm = LinearRegressor([tf.feature_column.numeric_column(m) for m in boston_data.columns])
lm.train(train_feed_input)

print(lm)
