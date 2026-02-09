import cv2
import numpy as np
from tensorflow import keras
import matplotlib.pyplot as plt

# Load trained model
model = keras.models.load_model("mnist_model.h5")

# Read image using OpenCV (grayscale)
img = cv2.imread("mydigit2.png", cv2.IMREAD_GRAYSCALE)

# Check if image loaded
if img is None:
    print("Error: Image not found!")
    exit()

# Resize to 28x28
img = cv2.resize(img, (28, 28))

# Invert colors (MNIST style)
img = 255 - img

# Normalize
img = img / 255.0

# Reshape for model
img = img.reshape(1, 28, 28)

# Predict
prediction = model.predict(img)
digit = np.argmax(prediction)

# Display image
plt.imshow(img.reshape(28,28), cmap="gray")
plt.title(f"Predicted: {digit}")
plt.axis("off")
plt.show()

print("Predicted Digit:", digit)
