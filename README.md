---

# Deepfake Detection Project

This project uses a Convolutional Neural Network (CNN) model to detect AI-generated images and classify them as either real or fake.
The dataset used is the CIFAKE 2023 dataset, which contains modern, high-quality images.

## Project Overview

This project aims to classify images as either real or AI-generated.
It demonstrates how to build and train a CNN model for deepfake image detection.

**Key points:**

* Input: Images resized to 32x32 pixels
* Output: Binary classification (Real / AI-generated)
* Dataset: CIFAKE 2023

## Dataset

* Source: Kaggle (birdy654/cifake-real-and-ai-generated-synthetic-images)
* Training images: 100,000 (50% real, 50% fake)
* Testing images: 20,000 (50% real, 50% fake)

## Requirements

Install the following Python packages before running the code:

```
pip install tensorflow keras matplotlib kagglehub
```

## Setup and Loading Data

```python
import kagglehub

path = kagglehub.dataset_download("birdy654/cifake-real-and-ai-generated-synthetic-images")
print("Path to dataset files:", path)
```

```python
import os

dataset_path = '/kaggle/input/cifake-real-and-ai-generated-synthetic-images'
train_dir = os.path.join(dataset_path, 'train')
test_dir = os.path.join(dataset_path, 'test')

if os.path.exists(train_dir) and os.path.exists(test_dir):
    print("Dataset Successfully loaded")
```

## Data Preprocessing

* Rescale pixel values between 0 and 1
* Apply data augmentation such as random rotation, shifting, shearing, zooming, and flipping
* Use `ImageDataGenerator` from Keras to prepare data for training and testing.

## Model Architecture

The CNN model consists of:

* 3 convolutional layers with max pooling
* A flatten layer
* A fully connected dense layer
* Dropout for regularization
* A sigmoid output layer for binary classification

The model is compiled with the Adam optimizer and binary cross-entropy loss.

## Training

* Epochs: 20
* Batch size: 64
* Trained using the CIFAKE dataset
* Validated on the test set

Training time may vary depending on hardware.

## Evaluation

The model achieved approximately 86% accuracy on the test set.

## Saving and Using the Model

The trained model is saved as:

```
real_vs_fake_image_classifier.h5
```

To make predictions on new images, load the model and use it to classify the image as either real or fake.

## Visualizing Training Performance

The training and validation accuracy and loss can be visualized using matplotlib to monitor the model’s performance and detect overfitting or underfitting.

## Future Work

* Improve model accuracy using transfer learning or more advanced architectures
* Add a user interface or web application for easier use
* Integrate real-time detection tools

## Summary

* Dataset: CIFAKE 2023 (real and AI-generated images)
* Model: CNN for binary image classification
* Purpose: Detect AI-generated images
* Accuracy: Around 86% on the test set
* Saved model name: `real_vs_fake_image_classifier.h5`
* Status: The project is still under development and will be improved in the future.
