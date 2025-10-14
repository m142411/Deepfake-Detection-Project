# Deepfake Detection Project

A Convolutional Neural Network (CNN) model trained to detect AI-generated images from the 2023 CIFAKE dataset.

## Project Overview

This project implements a CNN to classify images as **REAL** or **AI-generated (FAKE)**. The model is trained on a large dataset and demonstrates strong accuracy in distinguishing between real and synthetic images.

---

## Dataset

- **Source:** [CIFAKE 2023](https://www.kaggle.com/birdy654/cifake-real-and-ai-generated-synthetic-images)  
- **Training images:** 100,000  
- **Testing images:** 20,000  
- **Classes:** REAL, FAKE  
- **Note:** The dataset is recent (2023) and must be downloaded from Kaggle. It is **not included** in this repository due to size constraints.  

---

## Environment Setup

```bash
# Install required packages
pip install tensorflow keras matplotlib

