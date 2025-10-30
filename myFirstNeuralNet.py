import numpy as np

class NeuralNetwork:
    def __init__(self, layers, lr=0.05):
        self.lr = lr
        self.L = len(layers) - 1
        self.W = [np.random.randn(layers[i], layers[i+1]) * np.sqrt(2/layers[i]) for i in range(self.L)]
        self.b = [np.zeros((1, layers[i+1])) for i in range(self.L)]

    def sigmoid(self, Z):
        return 1 / (1 + np.exp(-Z))
    def sigmoid_derivative(self, A):
        return A * (1 - A)
    def relu(self, Z):
        return np.maximum(0, Z)
    def relu_derivative(self, A):
        return (A > 0).astype(float)

    def forward(self, X):
        self.Z, self.A = [], [X]
        for i in range(self.L - 1):
            Z = np.dot(self.A[-1], self.W[i]) + self.b[i]
            A = self.relu(Z)
            self.Z.append(Z)
            self.A.append(A)
        Z = np.dot(self.A[-1], self.W[-1]) + self.b[-1]
        A = self.sigmoid(Z)
        self.Z.append(Z)
        self.A.append(A)
        return A

    def backward(self, y):
        m = y.shape[0]
        grads_W, grads_b = [], []
        dA = self.A[-1] - y
        dZ = dA * self.sigmoid_derivative(self.A[-1])

        for i in reversed(range(self.L)):
            dW = (self.A[i].T @ dZ) / m
            dB = np.mean(dZ, axis=0, keepdims=True)
            grads_W.insert(0, dW)
            grads_b.insert(0, dB)
            if i > 0:
                dA = dZ @ self.W[i].T
                dZ = dA * self.relu_derivative(self.A[i])

        for i in range(self.L):
            self.W[i] -= self.lr * grads_W[i]
            self.b[i] -= self.lr * grads_b[i]

        grad_norms = [np.linalg.norm(g) for g in grads_W]
        return grad_norms

    def compute_statistics(self, A):
        mean = np.mean(A, axis=0)
        var = np.var(A, axis=0)
        cov = np.cov(A.T)
        return mean, var, cov

    def train(self, X, y, epochs=10000):
        for i in range(epochs):
            output = self.forward(X)
            loss = np.mean((y - output) ** 2)
            grad_norms = self.backward(y)
            if i % 2000 == 0:
                mean, var, _ = self.compute_statistics(self.A[-1])
                print(f"Epoch {i} | Loss: {loss:.4f}")
                print(f" Output mean: {mean.round(3)} | var: {var.round(3)} | grad norms: {[round(g,3) for g in grad_norms]}")

X = np.array([[0,0],[0,1],[1,0],[1,1]])
y = np.array([[0],[1],[1],[0]])

np.random.seed(42)
nn = NeuralNetwork([2, 6, 4, 3, 1], lr=0.05)
nn.train(X, y, 10000)
print("\nFinal predictions:")
print(nn.forward(X).round(3))