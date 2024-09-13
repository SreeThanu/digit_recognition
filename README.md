# Handwritten Digit Recognition using Convolutional Neural Networks (CNN)

This project demonstrates a deep learning model that recognizes handwritten digits from the famous [MNIST dataset](http://yann.lecun.com/exdb/mnist/). The model is built using Convolutional Neural Networks (CNNs), which are particularly effective for image-related tasks like digit classification.

## Overview

The goal of this project is to classify images of handwritten digits (0-9) into their corresponding categories using CNNs. CNNs are well-suited for image recognition tasks because of their ability to learn spatial hierarchies and features through multiple layers of convolution and pooling.

## Dataset

We use the MNIST dataset, which contains 70,000 grayscale images of handwritten digits. The dataset is split into:
- **Training Set**: 60,000 images
- **Test Set**: 10,000 images

Each image is of size 28x28 pixels and corresponds to a digit between 0 and 9.

## Model Architecture

The CNN architecture used in this project consists of multiple convolutional layers followed by pooling and fully connected layers. The key components are:

- **Convolutional Layers**: These layers apply filters to the input image to detect different features like edges, textures, and shapes.
- **Pooling Layers**: These layers down-sample the image, reducing its size and complexity while retaining the most important information.
- **Fully Connected Layers**: After several convolution and pooling operations, the output is flattened and passed through fully connected layers to perform the final classification.
- **Activation Function**: ReLU activation is used in the hidden layers, and softmax is used for the output layer to classify the digits.

## Steps

1. **Data Preprocessing**: The images are normalized and reshaped to fit the model input dimensions.
2. **Model Construction**: A CNN model is built with several layers of convolution, pooling, and fully connected layers.
3. **Training**: The model is trained using the training data for a fixed number of epochs.
4. **Evaluation**: The model's performance is evaluated on the test data.

## Dependencies

- Python 3.x
- TensorFlow / Keras
- NumPy
- Matplotlib
- Scikit-learn

To install the required dependencies, run:

```bash
pip install tensorflow numpy matplotlib scikit-learn



How to Run
Clone this repository:

bash
Copy code
git clone https://github.com/yourusername/digit_recognition.git
Navigate to the project directory:

bash
Copy code
cd digit_recognition
Run the Jupyter notebook:

bash
Copy code
jupyter notebook digit.ipynb
Follow the steps in the notebook to train and evaluate the model.
