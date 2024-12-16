# Plant Disease Detection and Data Augmentation

This project implements a Conditional Generative Adversarial Network (cGAN) for generating synthetic plant disease images and performs disease detection using Vision Transformers (ViT) and ResNet-50.

## Overview

The project consists of two main components:
1. A Conditional GAN for generating synthetic plant disease images
2. Vision Transformer (ViT) for plant disease classification
3. ResNet-50 for plant disease classification

### Conditional GAN Architecture

The cGAN implementation includes:
- **Generator**: Takes random noise and class conditions to generate synthetic plant disease images
- **Discriminator**: Evaluates whether images are real or generated
- **Image Resolution**: Generates 128x128 RGB images
- **Classes**: Handles three disease categories
  - Healthy
  - Powdery Mildew
  - Rust

Key features:
- Label smoothing for stable training
- Batch normalization in both generator and discriminator
- LeakyReLU activation in discriminator
- Tanh activation in generator output

### Training Details

- Training Parameters:
  - Epochs: 200
  - Learning Rate: 0.0001
  - Beta1: 0.5
  - Image Size: 128x128
  - Batch Size: 64
- GPU acceleration enabled
- Separate generators trained for each disease class
- BCE Loss for both generator and discriminator

## Requirements

- Python 3.10+
- PyTorch
- torchvision
- matplotlib
- CUDA-capable GPU (recommended)

## Dataset

The project uses the Plant Disease Dataset from Kaggle, which includes:
- 3 classes of plant diseases (healthy, powdery, and rust)
- RGB images of plant leaves
- Training, validation, and test splits