# MNIST Digit Classifier

This project is a handwritten digit recognition system developed using Python and TensorFlow. The model is trained on the MNIST dataset to classify digits from 0 to 9.

---

## Project Overview

The aim of this project is to build a neural network that can recognize handwritten digits from grayscale images. The system takes an image as input and predicts the digit present in it.

---

## Objective

- To understand the basics of machine learning and neural networks.
- To train a model on the MNIST dataset.
- To evaluate the model using test data.
- To test the model on custom handwritten images.

---

## Dataset

The MNIST dataset contains 60,000 training images and 10,000 testing images of handwritten digits. Each image is of size 28×28 pixels and is in grayscale format.

---

## Model Architecture

The neural network used in this project consists of:

- Flatten Layer (28×28 → 784 neurons)
- Dense Layer with 128 neurons and ReLU activation
- Output Layer with 10 neurons and Softmax activation

---

## Technologies Used

- Python 3.10
- TensorFlow and Keras
- NumPy
- Matplotlib
- OpenCV / Pillow (for custom image testing)

---

## How to Run the Project

1. Install the required libraries:

   ```bash
   pip install -r requirements.txt
