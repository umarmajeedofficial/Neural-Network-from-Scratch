# main.py
from network import NeuralNetwork
from utils import load_mnist_data

X_train, y_train, X_test, y_test = load_mnist_data()

model = NeuralNetwork([784, 128, 10])  # input layer, hidden, output
model.train(X_train, y_train, epochs=10, lr=0.01)

acc = model.evaluate(X_test, y_test)
print(f"Test Accuracy: {acc:.2f}%")
