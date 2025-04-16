# network.py
import numpy as np

class NeuralNetwork:
    def __init__(self, layers):
        self.weights = [np.random.randn(layers[i], layers[i+1]) * 0.1 for i in range(len(layers)-1)]
        self.biases = [np.zeros((1, layers[i+1])) for i in range(len(layers)-1)]

    def relu(self, x): return np.maximum(0, x)
    def softmax(self, x): e = np.exp(x - np.max(x, axis=1, keepdims=True)); return e / e.sum(axis=1, keepdims=True)
    def relu_deriv(self, x): return (x > 0).astype(float)

    def forward(self, X):
        activations = [X]
        Z = X
        for i in range(len(self.weights) - 1):
            Z = self.relu(Z @ self.weights[i] + self.biases[i])
            activations.append(Z)
        out = self.softmax(Z @ self.weights[-1] + self.biases[-1])
        activations.append(out)
        return activations

    def backward(self, X, y, activations, lr):
        y_onehot = np.zeros_like(activations[-1])
        y_onehot[np.arange(len(y)), y] = 1
        delta = activations[-1] - y_onehot

        for i in reversed(range(len(self.weights))):
            a_prev = activations[i]
            grad_w = a_prev.T @ delta
            grad_b = np.sum(delta, axis=0, keepdims=True)
            self.weights[i] -= lr * grad_w
            self.biases[i] -= lr * grad_b
            if i != 0:
                delta = (delta @ self.weights[i].T) * self.relu_deriv(a_prev)

    def train(self, X, y, epochs=10, lr=0.01):
        for epoch in range(epochs):
            activations = self.forward(X)
            self.backward(X, y, activations, lr)
            acc = self.evaluate(X, y)
            print(f"Epoch {epoch+1}, Accuracy: {acc:.2f}%")

    def evaluate(self, X, y):
        out = self.forward(X)[-1]
        preds = np.argmax(out, axis=1)
        return (preds == y).mean() * 100
