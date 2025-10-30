import numpy as np

class NeuralNetwork:
    def __init__(self, input_size, h1, h2, output_size, lr=0.1):
        self.W1 = np.random.randn(input_size, h1)
        self.b1 = np.zeros((1, h1))
        self.W2 = np.random.randn(h1, h2)
        self.b2 = np.zeros((1, h2))
        self.W3 = np.random.randn(h2, output_size)
        self.b3 = np.zeros((1, output_size))
        self.lr = lr

    def sigmoid(self, x):
        return 1 / (1 + np.exp(-x))
    def sigmoid_derivative(self, x):
        return x * (1 - x)
    def relu(self, x):
        return np.maximum(0, x)
    def relu_derivative(self, x):
        return (x > 0).astype(float)

    def forward(self, X):
        self.z1 = np.dot(X, self.W1) + self.b1
        self.a1 = self.relu(self.z1)
        self.z2 = np.dot(self.a1, self.W2) + self.b2
        self.a2 = self.relu(self.z2)
        self.z3 = np.dot(self.a2, self.W3) + self.b3
        self.a3 = self.sigmoid(self.z3)
        return self.a3

    def backward(self, X, y):
        d_a3 = self.a3 - y
        d_z3 = d_a3 * self.sigmoid_derivative(self.a3)
        d_W3 = np.dot(self.a2.T, d_z3)
        d_b3 = np.sum(d_z3, axis=0, keepdims=True)

        d_a2 = np.dot(d_z3, self.W3.T)
        d_z2 = d_a2 * self.relu_derivative(self.a2)
        d_W2 = np.dot(self.a1.T, d_z2)
        d_b2 = np.sum(d_z2, axis=0, keepdims=True)

        d_a1 = np.dot(d_z2, self.W2.T)
        d_z1 = d_a1 * self.relu_derivative(self.a1)
        d_W1 = np.dot(X.T, d_z1)
        d_b1 = np.sum(d_z1, axis=0, keepdims=True)

        self.W3 -= self.lr * d_W3; self.b3 -= self.lr * d_b3
        self.W2 -= self.lr * d_W2; self.b2 -= self.lr * d_b2
        self.W1 -= self.lr * d_W1; self.b1 -= self.lr * d_b1

    def train(self, X, y, epochs):
        for i in range(epochs):
            self.forward(X)
            loss = np.mean((y - self.a3) ** 2)
            self.backward(X, y)
            if i % 1000 == 0:
                print(f"Epoch {i}, Loss: {loss:.4f}")

X = np.array([[0,0],[0,1],[1,0],[1,1]])
y = np.array([[0],[1],[1],[0]])

np.random.seed(42)
nn = NeuralNetwork(2, 4, 3, 1, lr=0.1)
nn.train(X, y, 15000)
print(nn.forward(X).round(3))