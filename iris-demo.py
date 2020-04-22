import csv
import tensorflow as tf
import pandas as pd 
import matplotlib.pyplot as plt 
import seaborn as sns
import pytest


df = pd.read_csv("https://gist.githubusercontent.com/netj/8836201/raw/6f9306ad21398ea43cba4f7d537619d0e07d5ae3/iris.csv")

sns.scatterplot(x="petal.length", y="petal.width", hue="variety", data=df)

def test_csv_len():
    assert len(df) == 150
