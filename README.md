### ResNet50
A residual neural network (also referred to as a residual network or ResNet)[1] is a deep learning architecture in which the weight layers learn residual functions with reference to the layer inputs. It was developed in 2015 for image recognition and won that year's ImageNet Large Scale Visual Recognition Challenge : allows networks to go much deeper (up to hundreds of layers) while maintaining good training efficiency. The main difference in ResNets is that they have shortcut connections parallel to their normal convolutional layers. Contrary to convolution layers, these shortcut connections are always alive and the gradients can easily back propagate through them, which results in a faster training.

It is an innovative neural network created for image classification.

The ResNet model architecture allows the training error to be reduced with a deeper network through connection skip.

Residual neural networks ignore some connections and make double or triple layer jumps that contain non-linearities (ReLU)
![ResNet50](https://github.com/user-attachments/assets/f6337e4d-c832-4b48-a864-2d9b3a8b06ea)

## Transfer Learning
-If a model is trained on a database, there is no need to re-train the model from scratch to fit a new set of similar data.

saving resources
improving efficiency
model training facilitation
saving time
The task is to transfer the learning from a ResNet50, trained with Imagenet dataset, to a model that classifies flower images -Transfer learning is a machine learning technique where a model trained on one task is reused or fine-tuned for a different but related task. Instead of training a new model from scratch, transfer learning leverages the knowledge gained from a pre-trained model, allowing it to achieve better performance with less data and computation on the new task. This approach is especially effective when the new task has limited labeled data.

To show how Transfer Learning can be useful, ResNet50 will be trained on a custom dataset.

#### Use Case : Flower Classification
Flower types - daisy, dandelion, roses, sunflowers, tulips
To classify flower images, the Flower Classification dataset will be used. It is available on Kaggle : 
The dataset contains 5 types of flowers:

*daisy

*dandelion

*roses

*sunflowers

*tulips

### Model Architecture Description
###### Overview
The model is built using transfer learning with the ResNet50 architecture, a deep convolutional neural network known for its ability to learn complex features while mitigating the vanishing gradient problem through skip connections (also known as residual connections). The model is tailored to classify images of flowers into five distinct categories: daisies, dandelions, roses, sunflowers, and tulips.

###### Architecture Components
Base Model (ResNet50):

Input Layer: The model accepts input images of size 224x224x3 (height, width, channels), which is a common input size for many pretrained models.
Pretrained Weights: ResNet50 is initialized with weights pretrained on the ImageNet dataset, allowing it to leverage learned features from a large and diverse set of images. This helps improve performance and reduces training time.
Layers: ResNet50 consists of multiple convolutional layers arranged in blocks. Each block contains:
Convolutional layers that extract features from the input images.
Batch normalization layers that stabilize learning.
ReLU activation functions that introduce non-linearity.
Skip connections that allow gradients to flow more easily during backpropagation.
Global Average Pooling Layer:

After passing through the ResNet50 base, the output feature maps are fed into a Global Average Pooling layer. This layer reduces the spatial dimensions of the feature maps by averaging them, resulting in a single vector of features for each image. This helps minimize overfitting and reduces the model's complexity.
Fully Connected Layers:

The model includes several fully connected (dense) layers:
Dense Layer 1: 256 neurons with ReLU activation. This layer captures complex interactions between features learned from the ResNet50.
Dropout Layer 1: A dropout rate of 20% is applied to prevent overfitting by randomly dropping units during training.
Dense Layer 2: 128 neurons with ReLU activation, further learning complex patterns.
Dropout Layer 2: Another dropout layer with a 20% rate.
Dense Layer 3: 64 neurons with ReLU activation to learn finer details from the preceding layers.
Dropout Layer 3: A third dropout layer with a 20% rate.
Output Layer:

The final layer is a Dense Layer with Softmax Activation that outputs probabilities for each of the five classes (daisy, dandelion, rose, sunflower, tulip). This layer translates the learned features into class predictions.
