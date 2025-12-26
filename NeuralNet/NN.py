import numpy as np


def Activation(z, acitve):
    def sigmoid(z):
        return 1 / (1 + np.exp(-z))

    def sigmoid_derivative(a):
        return a * (1 - a)

    def relu(z):
        return np.maximum(0, z)

    def relu_derivative(z):
        return (z > 0).astype(float)

    def tangh(z):
        return np.tanh(z)
    
    def tangh_derivative(a):
        return 1 - a ** 2
    
    def softmax(z):
        exp_z = np.exp(z - np.max(z, axis=1, keepdims=True))
        return exp_z / np.sum(exp_z, axis=1, keepdims=True)
    def softmax_derivative(a):
        return a * (1 - a)
    
    dict_1 = {
        "sigmoid": sigmoid,
        'softmax': softmax,
        "relu": relu,
        "tangh": tangh,
        "tangh_derivative": tangh_derivative,
        "sigmoid_derivative": sigmoid_derivative,
        "relu_derivative": relu_derivative,
        "softmax_derivative": softmax_derivative
    }

    return dict_1[acitve](z)

class DenseLayer:
    def __init__(self, input_dim, output_dim, activation='relu', lr=0.05):
        self.activation = activation
        self.lr = lr
        self.X = None
        self.Z = None
        self.A = None

        if activation == 'relu':
            self.W = np.random.randn(input_dim, output_dim) * np.sqrt(2 / input_dim)
        elif activation in ['sigmoid', 'tangh']:
            self.W = np.random.randn(input_dim, output_dim) * np.sqrt(1 / input_dim)
        else:
            self.W = np.random.randn(input_dim, output_dim) * 0.01

        self.b = np.zeros((1, output_dim))

    def forward(self, X):
        self.X = X
        self.Z = X @ self.W + self.b
        self.A = Activation(self.Z, self.activation)
        return self.A

    def backward(self, dA):
        # для сигмоиды на выходе + BCE
        if self.activation == 'sigmoid':
            dZ = dA
        else:
            dZ = dA * Activation(self.Z, self.activation + '_derivative')

        dW = self.X.T @ dZ
        db = np.sum(dZ, axis=0, keepdims=True)
        dX = dZ @ self.W.T

        self.W -= self.lr * dW
        self.b -= self.lr * db

        return dX



class BCELoss:
    @staticmethod
    def forward(y_pred, y_true):
        eps = 1e-8
        return -np.mean(
            y_true * np.log(y_pred + eps) +
            (1 - y_true) * np.log(1 - y_pred + eps)
        )

    @staticmethod
    def backward(y_pred, y_true):
        return (y_pred - y_true) / y_true.shape[0]


class NeuralNetwork:
    def __init__(self):
        self.layers = []

    def add(self, layer):
        self.layers.append(layer)

    def forward(self, X):
        for layer in self.layers:
            X = layer.forward(X)
        return X

    def backward(self, dLoss):
        for layer in reversed(self.layers):
            dLoss = layer.backward(dLoss)

    def fit(self, X, y, epochs=100):
        for epoch in range(epochs):
            y_pred = self.forward(X)
            loss = BCELoss.forward(y_pred, y)
            dLoss = BCELoss.backward(y_pred, y)
            self.backward(dLoss)

            if epoch % 10 == 0:
                print(f"epoch {epoch} | loss {loss:.4f}")

    def predict(self, X):
        return self.forward(X)
