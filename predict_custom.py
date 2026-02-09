import numpy as np
from tensorflow import keras
from PIL import Image
import matplotlib.pyplot as plt

# Load trained model
model = keras.models.load_model("mnist_model.h5")

# Load and preprocess image
img = Image.open("mydigit.png").convert("L")   # grayscale
img = img.resize((28, 28))                     # resize

img_array = np.array(img)

# Invert colors (MNIST style)
img_array = 255 - img_array

# Normalize
img_array = img_array / 255.0

# Reshape
img_array = img_array.reshape(1, 28, 28)

# Predict
prediction = model.predict(img_array)
digit = np.argmax(prediction)

# Show result
plt.imshow(img_array.reshape(28,28), cmap="gray")
plt.title(f"Predicted: {digit}")
plt.axis("off")
plt.show()

print("Predicted Digit:", digit)
