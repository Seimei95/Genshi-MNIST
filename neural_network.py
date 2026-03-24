import numpy as np
from sklearn.datasets import fetch_openml

# loading dataset
mnist = fetch_openml("mnist_784", as_frame=False)
X_raw = mnist.data  # pixel values
y_raw = mnist.target  # labels
# converting output strings into integers
y_digits = y_raw.astype(np.int32)

# splitting without shuffling as 60:10
X_train_raw = X_raw[0:60000]
X_test_raw = X_raw[60000:]
y_train_digits = y_digits[:60000]
y_test_digits = y_digits[60000:]

# normalize the image by dividing with 255
X_train = (X_train_raw / 255.0).astype(np.float32)
X_test = (X_test_raw / 255.0).astype(np.float32)
# Label Conversion
y_train = (y_train_digits % 2 == 0).astype(np.float32).reshape(-1, 1)
y_test = (y_test_digits % 2 == 0).astype(np.float32).reshape(-1, 1)

print("X_train shape:", X_train.shape)  # (60000, 784)
print("X_test shape: ", X_test.shape)  # (10000, 784)
print("y_train shape:", y_train.shape)  # (60000, 1)
print("y_test shape: ", y_test.shape)  # (10000, 1)
print("y_train dtype:", y_train.dtype)  # float32
print("Sample labels:", y_train[:5].flatten())


def sigmoid(z):
    ## converting output strings into integers
    z = np.clip(z, -500, 500)
    return 1 / (1 + np.exp(-z))


def compute_loss(y_true, y_hat):
    eps = 1e-15
    y_hat = np.clip(y_hat, eps, 1 - eps)
    # binary-cross-entropy-loss function
    return -np.mean(y_true * np.log(y_hat) + (1 - y_true) * np.log(1 - y_hat))


# testing both functions
# Test sigmoid
print(sigmoid(0))  # should print 0.5
print(sigmoid(100))  # should print 1.0 (or very close)
print(sigmoid(-100))  # should print 0.0 (or very close)

# Test compute_loss
# Perfect predictions — loss should be tiny
y_true = np.array([[1.0], [0.0], [1.0]])
y_hat_perfect = np.array([[0.99], [0.01], [0.99]])
print(compute_loss(y_true, y_hat_perfect))  # should be very small ~0.01

# Terrible predictions — loss should be large
y_hat_terrible = np.array([[0.01], [0.99], [0.01]])
print(compute_loss(y_true, y_hat_terrible))  # should be very large ~4.6


def initialize_weights(hidden_size=64):
    W_to_H = np.random.randn(784, hidden_size).astype(np.float32) * 0.01
    B_to_H = np.zeros(hidden_size, dtype=np.float32)
    Wh_to_O = np.random.randn(hidden_size, 1).astype(np.float32) * 0.01
    B_to_O = np.zeros(1, dtype=np.float32)

    return W_to_H, B_to_H, Wh_to_O, B_to_O


W_to_H, B_to_H, Wh_to_O, B_to_O = initialize_weights(hidden_size=64)

print(W_to_H.shape)  # (784, 64)
print(B_to_H.shape)  # (64,)
print(Wh_to_O.shape)  # (64, 1)
print(B_to_O.shape)  # (1,)
print(W_to_H.dtype)  # float32
print(W_to_H.max())  # something small like 0.03
print(W_to_H.min())  # something small like -0.03
