# Flower Classification Using ResNet-50 and Transfer Learning

## ResNet50

A Residual Neural Network (ResNet) is a deep learning architecture developed in 2015 for image recognition, which won that year's ImageNet Large Scale Visual Recognition Challenge. It allows networks to go much deeper (up to hundreds of layers) while maintaining good training efficiency. The key innovation in ResNets is the introduction of shortcut connections that enable gradients to backpropagate easily, resulting in faster training.

![ResNet50](https://github.com/user-attachments/assets/f6337e4d-c832-4b48-a864-2d9b3a8b06ea)

## Transfer Learning

Transfer learning is a technique where a model trained on one task is reused or fine-tuned for a different but related task. This method allows us to:

- Save resources
- Improve efficiency
- Facilitate model training
- Save time

In this project, we transfer learning from a ResNet50 model trained on the ImageNet dataset to classify flower images. This approach is particularly effective when the new task has limited labeled data.

### Use Case: Flower Classification

The model will classify five types of flowers:

- Daisy
- Dandelion
- Roses
- Sunflowers
- Tulips

The dataset for this task is available on Kaggle and contains images of the above flower types.

## Model Architecture Description

### Overview

The model leverages transfer learning using the ResNet50 architecture, implemented in a Vertex AI instance within Jupyter Lab. ResNet50 is a deep convolutional neural network that addresses the vanishing gradient problem through skip connections (residual connections). This model is designed to classify flower images into five distinct categories: daisies, dandelions, roses, sunflowers, and tulips.

### Architecture Components

- **Base Model (ResNet50)**:
  - **Input Layer**: Accepts input images of size 224x224x3.
  - **Pretrained Weights**: Initialized with weights from the ImageNet dataset.
  - **Layers**: Comprises multiple convolutional layers arranged in blocks, including:
    - Convolutional layers for feature extraction.
    - Batch normalization layers for stabilizing learning.
    - ReLU activation functions to introduce non-linearity.
    - Skip connections to enhance gradient flow.

- **Global Average Pooling Layer**: Reduces spatial dimensions of feature maps by averaging them into a single vector for each image.

- **Fully Connected Layers**:
  - **Dense Layer 1**: 256 neurons with ReLU activation.
  - **Dropout Layer 1**: 20% dropout rate to prevent overfitting.
  - **Dense Layer 2**: 128 neurons with ReLU activation.
  - **Dropout Layer 2**: 20% dropout rate.
  - **Dense Layer 3**: 64 neurons with ReLU activation.
  - **Dropout Layer 3**: 20% dropout rate.
  - **Output Layer**: A Dense Layer with Softmax activation that outputs probabilities for each of the five classes.

## Getting Started

To set up the project locally, follow these steps:

### Clone the Repository

```bash
git clone https://github.com/malekzitouni/Flower-Classification-Using-ResNet-50-and-Transfer-Learning

### Install Requirements
pip install -r requirements.txt
### Download the Model
!pip install wget
!wget https://github.com/malekzitouni/Flower-Classification-Using-ResNet-50-and-Transfer-Learning/blob/main/model

### Conclusion
This project demonstrates how to use transfer learning with the ResNet50 architecture for flower classification. Feel free to explore, modify, and contribute to this repository!

###License

### Key Changes

1. **Structure**: Organized sections for better flow and readability.
2. **Clarity**: Simplified language and improved explanations.
3. **Markdown Formatting**: Enhanced formatting for better visual structure.
4. **Instructions**: Clearer steps for cloning the repository and downloading the model.

### Steps to Update Your README

1. **Edit the `README.md`**: Open the file in your project directory.
2. **Replace with the Updated Content**: Copy and paste the revised content.
3. **Save the Changes**.

### Commit and Push Changes

After updating the README:

```bash
git add README.md
git commit -m "Update README with project details and usage instructions"
git push origin main

