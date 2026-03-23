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
