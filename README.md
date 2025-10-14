# Deepfake Detection Project

A Convolutional Neural Network (CNN) model trained to detect AI-generated images from real ones using the **2023 CIFAKE dataset**, which contains modern, high-quality images.

---

## Project Overview

This project aims to **classify images as either real or AI-generated** (deepfake).  
It demonstrates the use of a CNN for image classification and deepfake detection.

**Key Points:**
- Input: Images (32x32 pixels)
- Output: Binary classification (Real / AI-generated)
- Dataset: Modern CIFAKE 2023 dataset

---

## Dataset

- **Source:** Kaggle (`birdy654/cifake-real-and-ai-generated-synthetic-images`)  
- **Training Images:** 100,000 (50% real, 50% AI-generated)  
- **Testing Images:** 20,000 (50% real, 50% AI-generated)  

> The dataset is recent and contains high-quality real and AI-generated images.

---

## Requirements

Python packages required:

```bash
pip install tensorflow keras matplotlib kagglehub
---

## Setup & Loading Data

1. Download dataset from Kaggle:

```python
import kagglehub

path = kagglehub.dataset_download("birdy654/cifake-real-and-ai-generated-synthetic-images")
print("Path to dataset files:", path)
```

2. Set paths for training and testing folders:

```python
import os

dataset_path = '/kaggle/input/cifake-real-and-ai-generated-synthetic-images'
train_dir = os.path.join(dataset_path, 'train')
test_dir = os.path.join(dataset_path, 'test')
```

3. Verify dataset loaded successfully:

```python
if os.path.exists(train_dir) and os.path.exists(test_dir):
    print("Dataset Successfully loaded")
```

---

## Data Preprocessing

Images are **rescaled and augmented** to improve model generalization:

* Rescale pixel values (0-1)
* Random rotations, shifts, shearing, zoom, and flips

This is done using `ImageDataGenerator` from Keras.

---

## Model Architecture

A simple **CNN** is used:

* **3 Convolutional layers** with MaxPooling
* Flatten layer
* Fully connected Dense layer
* Dropout for regularization
* Output layer with sigmoid activation for binary classification

The model is compiled with **Adam optimizer** and **binary cross-entropy loss**.

---

## Training

* Epochs: 20
* Batch size: 64

The model is trained on the CIFAKE dataset and validated on the test set.

> Training may take some time depending on your hardware.

---

## Evaluation

After training, the model achieves ~**86% accuracy** on the test set.

---

## Saving & Using the Model

You can save the trained model:

```python
model.save('real_vs_fake_image_classifier.h5')
```

And make predictions on new images:

```python
# Example:
# Image is classified as Real or AI-generated
```

---

## Visualizing Training Performance

The training and validation accuracy/loss can be visualized using matplotlib to check for overfitting or underfitting.

---

## Summary

* **Dataset:** Modern CIFAKE 2023 (real & AI-generated images)
* **Model:** CNN for binary image classification
* **Purpose:** Detect AI-generated images
* **Accuracy:** ~86% on test set

This project provides a complete pipeline from **loading modern image data**, **preprocessing**, **training a CNN**, **evaluating**, and **saving the model for predictions**.

