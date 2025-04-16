# utils.py
import numpy as np
from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split

def load_mnist_data():
    mnist = fetch_openml("mnist_784", version=1)
    X = mnist.data / 255.0
    y = mnist.target.astype(int)
    return train_test_split(X, y, test_size=0.2, random_state=42)
