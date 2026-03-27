import json
import platform
import resource

import matplotlib.pyplot as plt
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
    eps = 1e-7
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


def sigmoid(z):
    z = np.clip(z, -500, 500)
    return 1 / (1 + np.exp(-z))


def forward(X, W1, b1, W2, b2):
    # Hidden layer linear transformation
    Z1 = X @ W1 + b1
    # Hidden layer activation
    A1 = np.tanh(Z1)
    # Output layer linear transformation
    Z2 = A1 @ W2 + b2
    # Output activation
    y_hat = sigmoid(Z2)
    # save intermediate values for backpropagation
    cache = (Z1, A1, Z2)

    return y_hat, cache


# testing
# Make fake data to test with
X_fake = np.random.randn(5, 784).astype(np.float32)  # 5 fake images
W1, b1, W2, b2 = initialize_weights(hidden_size=64)

# Run forward pass
y_hat, cache = forward(X_fake, W1, b1, W2, b2)

# Check outputs
print(y_hat.shape)  # (5, 1)   ← 5 predictions, one per image
print(y_hat)  # all values between 0 and 1
print(y_hat.max())  # less than 1.0
print(y_hat.min())  # greater than 0.0

# Check cache
Z1, A1, Z2 = cache
print(Z1.shape)  # (5, 64)
print(A1.shape)  # (5, 64)
print(Z2.shape)  # (5, 1)


def backward(X, y, y_hat, cache, W2):
    n = X.shape[0]
    Z1, A1, Z2 = cache
    # Output error
    delta2 = y_hat - y

    dW2 = (1 / n) * A1.T @ delta2
    db2 = (1 / n) * np.sum(delta2, axis=0)

    delta1 = (delta2 @ W2.T) * (1 - A1**2)

    dW1 = (1 / n) * X.T @ delta1
    db1 = (1 / n) * np.sum(delta1, axis=0)

    grads = {"dW1": dW1, "db1": db1, "dW2": dW2, "db2": db2}
    return grads


# testing
X_fake = np.random.randn(5, 784).astype(np.float32)
y_fake = np.array([[1.0], [0.0], [1.0], [0.0], [1.0]])

W1, b1, W2, b2 = initialize_weights(hidden_size=64)

# Forward pass first
y_hat, cache = forward(X_fake, W1, b1, W2, b2)

# Now backward pass
grads = backward(X_fake, y_fake, y_hat, cache, W2)

# Check every gradient shape matches its weight shape
print(grads["dW1"].shape)  # (784, 64)  ← matches W1
print(grads["db1"].shape)  # (64,)      ← matches b1
print(grads["dW2"].shape)  # (64, 1)    ← matches W2
print(grads["db2"].shape)  # (1,)       ← matches b2


def gradient_check(X, y, W1, b1, W2, b2, param_name, i, j=None, epsilon=1e-7):

    X = X.astype(np.float64)
    y = y.astype(np.float64)
    W1 = W1.astype(np.float64)
    b1 = b1.astype(np.float64)
    W2 = W2.astype(np.float64)
    b2 = b2.astype(np.float64)
    # Get analytical gradient from backprop
    y_hat, cache = forward(X, W1, b1, W2, b2)
    grads = backward(X, y, y_hat, cache, W2)

    # pick the right gradient based on param_name
    if param_name == "W1":
        analytical = grads["dW1"][i, j]
    elif param_name == "W2":
        analytical = grads["dW2"][i, j]
    elif param_name == "b1":
        analytical = grads["db1"][i]
    elif param_name == "b2":
        analytical = grads["db2"][i]

    # Get numerical gradient by nudging
    # pick the right parameter to nudge
    if param_name == "W1":
        param = W1
    elif param_name == "W2":
        param = W2
    elif param_name == "b1":
        param = b1
    elif param_name == "b2":
        param = b2

    # save the original value
    if j is not None:
        original = param[i, j]  # for 2D arrays like W1, W2
    else:
        original = param[i]  # for 1D arrays like b1, b2

    # nudge UP and compute loss
    if j is not None:
        param[i, j] = original + epsilon
    else:
        param[i] = original + epsilon
    y_hat_plus, _ = forward(X, W1, b1, W2, b2)
    loss_plus = compute_loss(y, y_hat_plus)

    # nudge DOWN and compute loss
    if j is not None:
        param[i, j] = original - epsilon
    else:
        param[i] = original - epsilon
    y_hat_minus, _ = forward(X, W1, b1, W2, b2)
    loss_minus = compute_loss(y, y_hat_minus)

    # RESTORE original value — critical!
    if j is not None:
        param[i, j] = original
    else:
        param[i] = original

    # ── STEP 3: Compute numerical gradient ───────────────────
    numerical = (loss_plus - loss_minus) / (2 * epsilon)

    # ── STEP 4: Compare ──────────────────────────────────────
    difference = abs(analytical - numerical)

    return analytical, numerical, difference


# Use small data for speed — gradient check is slow on 60k samples
X_small = X_train[:100]
y_small = y_train[:100]

W1, b1, W2, b2 = initialize_weights(hidden_size=64)

# Check the 4 parameters the spec requires
a, n, d = gradient_check(X_small, y_small, W1, b1, W2, b2, "W1", 100, 30)
print(f"W1[100,30]  analytical={a:.2e}  numerical={n:.2e}  diff={d:.2e}")

a, n, d = gradient_check(X_small, y_small, W1, b1, W2, b2, "W1", 500, 50)
print(f"W1[500,50]  analytical={a:.2e}  numerical={n:.2e}  diff={d:.2e}")

a, n, d = gradient_check(X_small, y_small, W1, b1, W2, b2, "W2", 30, 0)
print(f"W2[30,0]    analytical={a:.2e}  numerical={n:.2e}  diff={d:.2e}")

a, n, d = gradient_check(X_small, y_small, W1, b1, W2, b2, "b1", 10)
print(f"b1[10]      analytical={a:.2e}  numerical={n:.2e}  diff={d:.2e}")


def train(X, y, hidden_size=64, learning_rate=0.1, iterations=1000):

    # ── STEP 1: Initialize weights ───────────────────────────
    W1, b1, W2, b2 = initialize_weights(hidden_size)

    losses = []  # to store loss at each iteration

    # ── STEP 2: Training loop ────────────────────────────────
    for i in range(iterations):
        # Forward pass — get predictions
        y_hat, cache = forward(X, W1, b1, W2, b2)

        # Compute loss — how wrong are we?
        loss = compute_loss(y, y_hat)
        losses.append(loss)

        # Backward pass — compute gradients
        grads = backward(X, y, y_hat, cache, W2)

        # Update weights — adjust in direction that reduces loss
        W1 = W1 - learning_rate * grads["dW1"]
        b1 = b1 - learning_rate * grads["db1"]
        W2 = W2 - learning_rate * grads["dW2"]
        b2 = b2 - learning_rate * grads["db2"]

        # Print progress every 100 iterations
        if i % 100 == 0:
            print(f"iteration {i}, loss = {loss:.4f}")

    return W1, b1, W2, b2, losses


def predict(X, W1, b1, W2, b2):
    # Run forward pass and convert probabilities to 0/1
    y_hat, _ = forward(X, W1, b1, W2, b2)
    return (y_hat > 0.5).astype(np.float32)


def accuracy(X, y, W1, b1, W2, b2):
    predictions = predict(X, W1, b1, W2, b2)
    return np.mean(predictions == y)


# Train with learning rate 0.1
print("Training with learning rate 0.1...")
W1, b1, W2, b2, losses = train(
    X_train, y_train, hidden_size=64, learning_rate=0.1, iterations=1000
)

# Check final loss
print(f"\nFinal loss: {losses[-1]:.4f}")

# Check accuracy on training set
train_acc = accuracy(X_train, y_train, W1, b1, W2, b2)
print(f"Train accuracy: {train_acc:.4f}")

# Check accuracy on test set
test_acc = accuracy(X_test, y_test, W1, b1, W2, b2)
print(f"Test accuracy:  {test_acc:.4f}")

plt.figure(figsize=(10, 5))
plt.plot(losses, label="learning rate = 0.1")
plt.xlabel("Iteration")
plt.ylabel("Loss")
plt.title("Training Loss Over Time")
plt.legend()
plt.grid(True)
plt.savefig("learning_curve.png")  # saves the plot as an image
plt.show()

# Train with learning rate 1.0
print("Training with learning rate 1.0...")
W1_b, b1_b, W2_b, b2_b, losses_2 = train(
    X_train, y_train, hidden_size=64, learning_rate=1.0, iterations=1000
)
print(f"\nFinal loss: {losses_2[-1]:.4f}")

train_acc_2 = accuracy(X_train, y_train, W1_b, b1_b, W2_b, b2_b)
print(f"Train accuracy: {train_acc_2:.4f}")

test_acc_2 = accuracy(X_test, y_test, W1_b, b1_b, W2_b, b2_b)
print(f"Test accuracy:  {test_acc_2:.4f}")

# Plot both on same graph
plt.figure(figsize=(10, 5))
plt.plot(losses, label="lr = 0.1")
plt.plot(losses_2, label="lr = 1.0")
plt.xlabel("Iteration")
plt.ylabel("Loss")
plt.title("Training Loss — Learning Rate Comparison")
plt.legend()
plt.grid(True)
plt.savefig("lr_comparison.png")
plt.show()

# ── PHASE 8: Hidden size experiments ─────────────────────────

all_losses = {"64": losses_2}  # already have this from lr=1.0 run

for hidden_size in [128, 256]:
    print(f"\nTraining with hidden_size={hidden_size}...")
    W1_h, b1_h, W2_h, b2_h, losses_h = train(
        X_train, y_train, hidden_size=hidden_size, learning_rate=1.0, iterations=1000
    )

    train_acc_h = accuracy(X_train, y_train, W1_h, b1_h, W2_h, b2_h)
    test_acc_h = accuracy(X_test, y_test, W1_h, b1_h, W2_h, b2_h)

    print(f"Final loss:     {losses_h[-1]:.4f}")
    print(f"Train accuracy: {train_acc_h:.4f}")
    print(f"Test accuracy:  {test_acc_h:.4f}")

    all_losses[str(hidden_size)] = losses_h


# ── Plot all three hidden sizes on one graph ──────────────────

plt.figure(figsize=(10, 5))
plt.plot(all_losses["64"], label="hidden size = 64")
plt.plot(all_losses["128"], label="hidden size = 128")
plt.plot(all_losses["256"], label="hidden size = 256")
plt.xlabel("Iteration")
plt.ylabel("Loss")
plt.title("Training Loss — Hidden Size Comparison")
plt.legend()
plt.grid(True)
plt.savefig("hidden_size_comparison.png")
plt.show()


def get_peak_memory_mb():
    if platform.system() == "Windows":
        import psutil

        return psutil.Process().memory_info().rss / (1024 * 1024)
    else:
        import resource

        usage = resource.getrusage(resource.RUSAGE_SELF)
        return usage.ru_maxrss / (1024 * 1024)


# Measure BEFORE training
memory_before = get_peak_memory_mb()

# Train
W1, b1, W2, b2, losses = train(
    X_train, y_train, hidden_size=64, learning_rate=1.0, iterations=1000
)

# Measure AFTER training
memory_after = get_peak_memory_mb()

# The difference is what training actually used
memory_used = memory_after - memory_before
print(f"Memory used by training: {memory_used:.1f} MB")
# Add after each train() call like this:
print(f"Peak memory: {get_peak_memory_mb():.1f} MB")
# Hidden size 64
memory_before = get_peak_memory_mb()
W1, b1, W2, b2, losses_64 = train(
    X_train, y_train, hidden_size=64, learning_rate=1.0, iterations=1000
)
mem_64 = get_peak_memory_mb() - memory_before
train_acc_64 = accuracy(X_train, y_train, W1, b1, W2, b2)
test_acc_64 = accuracy(X_test, y_test, W1, b1, W2, b2)
print(
    f"h=64  loss={losses_64[-1]:.4f}  train={train_acc_64:.4f}  test={test_acc_64:.4f}  mem={mem_64:.1f}MB"
)

# Hidden size 128
memory_before = get_peak_memory_mb()
W1, b1, W2, b2, losses_128 = train(
    X_train, y_train, hidden_size=128, learning_rate=1.0, iterations=1000
)
mem_128 = get_peak_memory_mb() - memory_before
train_acc_128 = accuracy(X_train, y_train, W1, b1, W2, b2)
test_acc_128 = accuracy(X_test, y_test, W1, b1, W2, b2)
print(
    f"h=128 loss={losses_128[-1]:.4f}  train={train_acc_128:.4f}  test={test_acc_128:.4f}  mem={mem_128:.1f}MB"
)

# Hidden size 256
memory_before = get_peak_memory_mb()
W1, b1, W2, b2, losses_256 = train(
    X_train, y_train, hidden_size=256, learning_rate=1.0, iterations=1000
)
mem_256 = get_peak_memory_mb() - memory_before
train_acc_256 = accuracy(X_train, y_train, W1, b1, W2, b2)
test_acc_256 = accuracy(X_test, y_test, W1, b1, W2, b2)
print(
    f"h=256 loss={losses_256[-1]:.4f}  train={train_acc_256:.4f}  test={test_acc_256:.4f}  mem={mem_256:.1f}MB"
)


# Add this anywhere in your code — just prints the table
def print_memory_table(hidden_size=64):
    bytes_per_element = 4  # float32

    arrays = {
        "W1": (784, hidden_size),
        "b1": (hidden_size,),
        "W2": (hidden_size, 1),
        "b2": (1,),
        "X_train": (60000, 784),
        "y_train": (60000, 1),
        "Z1": (60000, hidden_size),
        "A1": (60000, hidden_size),
    }

    print(f"\nMemory table (hidden_size={hidden_size}, float32):")
    print(f"{'Array':<12} {'Shape':<20} {'Elements':>12} {'MB':>10}")
    print("-" * 56)

    total = 0
    for name, shape in arrays.items():
        elements = 1
        for s in shape:
            elements *= s
        mb = elements * bytes_per_element / (1024 * 1024)
        total += mb
        shape_str = str(shape)
        print(f"{name:<12} {shape_str:<20} {elements:>12,} {mb:>10.4f}")

    print("-" * 56)
    print(f"{'TOTAL':<12} {'':<20} {'':<12} {total:>10.4f}")


print_memory_table(64)
print_memory_table(128)
print_memory_table(256)

results = {
    "learning_rates_tested": [0.1, 1.0],
    "hidden_sizes_tested": [64, 128, 256],
    "results": {
        "64": {
            "train_loss": round(float(losses_64[-1]), 4),
            "train_acc": round(float(train_acc_64), 4),
            "test_acc": round(float(test_acc_64), 4),
            "peak_memory_mb": round(float(mem_64), 1),
        },
        "128": {
            "train_loss": round(float(losses_128[-1]), 4),
            "train_acc": round(float(train_acc_128), 4),
            "test_acc": round(float(test_acc_128), 4),
            "peak_memory_mb": round(float(mem_128), 1),
        },
        "256": {
            "train_loss": round(float(losses_256[-1]), 4),
            "train_acc": round(float(train_acc_256), 4),
            "test_acc": round(float(test_acc_256), 4),
            "peak_memory_mb": round(float(mem_256), 1),
        },
    },
    "gradient_check": {
        "W1_100_30": 1.09e-09,
        "W1_500_50": 3.19e-10,
        "W2_30_0": 2.83e-11,
        "b1_10": 3.85e-10,
    },
}

with open("results.json", "w") as f:
    json.dump(results, f, indent=4)

print("results.json saved!")
print(json.dumps(results, indent=4))
