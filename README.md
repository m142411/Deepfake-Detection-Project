تمام، هذه نسخة README جاهزة للنسخ مباشرة على GitHub، كلها بصيغة Markdown، مرتبة وجاهزة:

````markdown
# Deepfake Detection Project

A Convolutional Neural Network (CNN) model trained to detect AI-generated images from real ones. This project uses the **2023 CIFAKE dataset**, which contains modern, high-quality real and AI-generated synthetic images.

---

## Table of Contents

1. [Project Description](#project-description)  
2. [Dataset](#dataset)  
3. [Requirements](#requirements)  
4. [Setup](#setup)  
5. [Data Preprocessing](#data-preprocessing)  
6. [Model Architecture](#model-architecture)  
7. [Training](#training)  
8. [Evaluation](#evaluation)  
9. [Saving the Model](#saving-the-model)  
10. [Making Predictions](#making-predictions)  
11. [Training History Visualization](#training-history-visualization)  

---

## Project Description

This project implements a CNN to classify images as either **real** or **AI-generated**. The goal is to automatically detect deepfake images using image classification techniques.

---

## Dataset

The dataset used in this project is from **CIFAKE 2023**, a modern dataset containing:  

- **100,000 training images** (50% real, 50% AI-generated)  
- **20,000 testing images** (50% real, 50% AI-generated)  

The dataset was downloaded directly from Kaggle using the `kagglehub` library.  

---

## Requirements

Install the required Python packages:

```bash
pip install tensorflow keras matplotlib kagglehub
````

---

## Setup

```python
import kagglehub

# Download latest version of CIFAKE dataset
path = kagglehub.dataset_download("birdy654/cifake-real-and-ai-generated-synthetic-images")

print("Path to dataset files:", path)
```

Set dataset paths:

```python
import os

dataset_path = '/kaggle/input/cifake-real-and-ai-generated-synthetic-images'
train_dir = os.path.join(dataset_path, 'train')
test_dir = os.path.join(dataset_path, 'test')

if not os.path.exists(train_dir) or not os.path.exists(test_dir):
    print("Dataset paths do not exist.")
else:
    print("Dataset Successfully loaded")
```

---

## Data Preprocessing

Images are preprocessed using `ImageDataGenerator`:

```python
from tensorflow.keras.preprocessing.image import ImageDataGenerator

IMG_HEIGHT = 32
IMG_WIDTH = 32
BATCH_SIZE = 64

train_datagen = ImageDataGenerator(
    rescale=1.0/255.0,
    rotation_range=20,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    fill_mode='nearest'
)

test_datagen = ImageDataGenerator(rescale=1.0/255.0)

train_generator = train_datagen.flow_from_directory(
    train_dir,
    target_size=(IMG_HEIGHT, IMG_WIDTH),
    batch_size=BATCH_SIZE,
    class_mode='binary'
)

test_generator = test_datagen.flow_from_directory(
    test_dir,
    target_size=(IMG_HEIGHT, IMG_WIDTH),
    batch_size=BATCH_SIZE,
    class_mode='binary'
)
```

---

## Model Architecture

A simple CNN model is used:

```python
from tensorflow.keras import layers, models

def build_model():
    model = models.Sequential()
    model.add(layers.Conv2D(32, (3, 3), activation='relu', input_shape=(IMG_HEIGHT, IMG_WIDTH, 3)))
    model.add(layers.MaxPooling2D((2, 2)))
    
    model.add(layers.Conv2D(64, (3, 3), activation='relu'))
    model.add(layers.MaxPooling2D((2, 2)))
    
    model.add(layers.Conv2D(128, (3, 3), activation='relu'))
    model.add(layers.MaxPooling2D((2, 2)))
    
    model.add(layers.Flatten())
    model.add(layers.Dense(128, activation='relu'))
    model.add(layers.Dropout(0.5))
    model.add(layers.Dense(1, activation='sigmoid'))
    
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    return model

model = build_model()
model.summary()
```

---

## Training

```python
EPOCHS = 20

history = model.fit(
    train_generator,
    epochs=EPOCHS,
    validation_data=test_generator
)
```

> Note: Training may take several hours depending on your hardware.

---

## Evaluation

Evaluate the model on the test set:

```python
test_loss, test_acc = model.evaluate(test_generator)
print(f'Test accuracy of Model is: {test_acc}')
```

---

## Saving the Model

You can save the trained model for later use:

```python
model.save('real_vs_fake_image_classifier.h5')
```

---

## Making Predictions

Predict if an image is real or AI-generated:

```python
import numpy as np
from tensorflow.keras.preprocessing import image

def predict_img(img_path, model):
    img = image.load_img(img_path, target_size=(IMG_HEIGHT, IMG_WIDTH))
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array /= 255.0

    prediction = model.predict(img_array)
    if prediction[0] > 0.5:
        print("Image is Real")
    else:
        print("Image is AI Generated")
```

---

## Training History Visualization

```python
import matplotlib.pyplot as plt

acc = history.history['accuracy']
val_acc = history.history['val_accuracy']
loss = history.history['loss']
val_loss = history.history['val_loss']

epochs_range = range(EPOCHS)

plt.figure(figsize=(12, 8))

plt.subplot(1, 2, 1)
plt.plot(epochs_range, acc, label='Training Accuracy')
plt.plot(epochs_range, val_acc, label='Validation Accuracy')
plt.legend(loc='lower right')
plt.title('Training and Validation Accuracy')

plt.subplot(1, 2, 2)
plt.plot(epochs_range, loss, label='Training Loss')
plt.plot(epochs_range, val_loss, label='Validation Loss')
plt.legend(loc='upper right')
plt.title('Training and Validation Loss')

plt.show()
```



إذا تحبين، أقدر أسوي لك نسخة **مختصرة وجاهزة للعرض مباشرة على GitHub** بحيث تكون صفحة الـ README قصيرة وواضحة، بدون كود كامل لكن مع روابط للملفات والكود.  

هل أسوي لك النسخة المختصرة؟
```
