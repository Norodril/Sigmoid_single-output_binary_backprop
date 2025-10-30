import numpy as np
import matplotlib.pyplot as plt

class NeuralNetwork:
    def __init__(self, layer_sizes, lr=0.05):
        self.lr = lr
        self.L = len(layer_sizes) - 1
        self.W = [np.random.randn(layer_sizes[i], layer_sizes[i+1]) * np.sqrt(2/layer_sizes[i]) for i in range(self.L)]
        self.b = [np.zeros((1, layer_sizes[i+1])) for i in range(self.L)]

    def sigmoid(self, x):
        return 1 / (1 + np.exp(-x))
    def sigmoid_derivative(self, x):
        return x * (1 - x)
    def relu(self, x):
        return np.maximum(0, x)
    def relu_derivative(self, x):
        return (x > 0).astype(float)

    def forward(self, X):
        self.z, self.a = [], [X]
        for i in range(self.L - 1):
            z = np.dot(self.a[-1], self.W[i]) + self.b[i]
            a = self.relu(z)
            self.z.append(z); self.a.append(a)
        z = np.dot(self.a[-1], self.W[-1]) + self.b[-1]
        a = self.sigmoid(z)
        self.z.append(z); self.a.append(a)
        return a

    def backward(self, y):
        m = y.shape[0]
        d_a = self.a[-1] - y
        d_z = d_a * self.sigmoid_derivative(self.a[-1])
        for i in reversed(range(self.L)):
            d_W = np.dot(self.a[i].T, d_z) / m
            d_b = np.sum(d_z, axis=0, keepdims=True) / m
            if i > 0:
                d_a = np.dot(d_z, self.W[i].T)
                d_z = d_a * self.relu_derivative(self.a[i])
            self.W[i] -= self.lr * d_W
            self.b[i] -= self.lr * d_b

    def train(self, X, y, epochs=10000):
        losses = []
        for i in range(epochs):
            output = self.forward(X)
            loss = np.mean((y - output) ** 2)
            self.backward(y)
            losses.append(loss)
            if i % 1000 == 0:
                print(f"Epoch {i}, Loss: {loss:.4f}")
        plt.plot(losses)
        plt.title("Training Loss Curve")
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.show()

X = np.array([[0,0],[0,1],[1,0],[1,1]])
y = np.array([[0],[1],[1],[0]])

np.random.seed(42)
nn = NeuralNetwork([2, 6, 4, 3, 1], lr=0.05)
nn.train(X, y, 15000)
print(nn.forward(X).round(3))