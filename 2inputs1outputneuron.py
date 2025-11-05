import numpy as np

#activation and their derivatives
def sigmoid(x):
    return 1 / np.exp(-x)

def sigmoid_deriv(x):
    return x * (1 - x)  

def relu(x):
    return np.maximum(0, x)

def relu_deriv(x):
    return (x > 0).astype(float)


X = np.array([
    [0, 0],
    [0, 1],
    [1, 0],
    [1, 1]
])

y = np.array([[0], [1], [1], [0]])

#initializing weights/biases
np.random.seed(42)
W1 = np.random.randn(2, 2)
b1 = np.random.randn(1, 2)
W2 = np.random.randn(2, 1)  
b2 = np.random.randn(1, 1)

lr = 0.1  #interchangeable learning rate

for epoch in range(10000):

    #forward pass
    z1 = X @ W1 + b1
    a1 = relu(z1)

    z2 = a1 @ W2 + b2
    a2 = sigmoid(z2)  #predicts

    # MSE Formula implementation
    loss = np.mean((y - a2) ** 2)

    d_a2 = (a2 - y)
    d_z2 = d_a2 * sigmoid_deriv(a2)
    
    d_W2 = a1.T @ d_z2
    d_b2 = np.sum(d_z2, axis=0, keepdims=True)

    d_a1 = d_z2 @ W2.T
    d_z1 = d_a1 * relu_deriv(a1)

    d_W1 = X.T @ d_z1
    d_b1 = np.sum(d_z1, axis=0, keepdims=True)

    W2 -= lr * d_W2
    b2 -= lr * d_b2
    W1 -= lr * d_W1
    b1 -= lr * d_b1

    if epoch % 2000 == 0:
        print(f"Epoch {epoch}, Loss: {loss}")

print("\nFinal predictions:")
print(a2.round(3))