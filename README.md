# Real-Time Gesture Recognition
Project for the Computer Vision 2023-2024 course at Sapienza

## Team

| **Name / Surname** | **Linkedin** | **GitHub** |
| :---: | :---: | :---: |
| `Bernardo Perrone De Menezes Bulcao Ribeiro ` | [![name](https://github.com/b-rbmp/NexxGate/blob/main/docs/logos/linkedin.png)](https://www.linkedin.com/in/b-rbmp/) | [![name](https://github.com/b-rbmp/NexxGate/blob/main/docs/logos/github.png)](https://github.com/b-rbmp) |
| `Roberta Chissich ` | [![name](https://github.com/b-rbmp/NexxGate/blob/main/docs/logos/linkedin.png)](https://www.linkedin.com/in/roberta-chissich/) | [![name](https://github.com/b-rbmp/NexxGate/blob/main/docs/logos/github.png)](https://github.com/RobCTs) |


## Table of Contents
+ [Introduction](#intro)
+ [Project Description](#project)
+ [System Components](#architecture)
+ [App/game](#app)
+ [Conclusions](#conclusion)
+ [Demo](#demo)
+ [References](#references)


## Introduction <a name = "intro"></a>

Hand Gesture Recognition is a technology that enables computers to interpret human gestures as commands. It works by capturing and analyzing the movements of the hands to recognize specific gestures and translate them into actions. This technology has a wide range of applications, from robotics and gaming to assistive technologies and human-computer interaction. Static Hand Gesture Recognition is a subset of this technology that focuses on recognizing static hand poses or shapes, which can be used to classify different gestures and commands based on a frame-by-frame analysis of the hand's position and shape.

### Importance and Applications 

One of the main applications of Hand Gesture Recognition is Human-Robot Interaction and Human Computer Interaction. By using hand gestures as commands, users can control robots, computers, and other devices in a more intuitive and natural way. This technology is particularly useful in scenarios where traditional input devices like keyboards and mice are not practical or feasible, such as in robotics, virtual reality, and augmented reality applications. Hand Gesture Recognition can also be used in healthcare, gaming, and security applications, among others.

## Project Description <a name = "project"></a>

### Objective
The primary objective of this project is to develop a real-time gesture recognition system that allows users to play a game of Rock-Paper-Scissors against a computer. The system will capture live video feeds of the user's hand gestures using a camera, analyze the movements to recognize the gestures using Deep Convolutional Neural Networks framed as a classification task, and translate them into commands to play the game. The computer will generate its own gestures randomly, and the system will determine the winner based on the rules of the game.

### Key Goals
**Motion Detection**: Utilize a standard smartphone camera to capture hand movements and translate them into commands.  
**Deep CNN Classification**: Implement a Deep Convolutional Neural Network (CNN) to classify hand gestures in real-time.
**Comparison of Models**: Compare the performance of different CNN models for gesture recognition, as well as the impact of data augmentation techniques.
**Mobile Application**: Implement a simple and intuitive mobile application that allows users to play the game using their smartphones.


## System components <a name = "architecture"></a>

### Model Architectures
**VGG16**
The VGG16 model is a convolutional neural network (CNN) consisting of 13 convolutional layers followed by 3 fully connected layers. This architecture is known for its simplicity and uniformity in the size of the convolution filters. The network has five max-pooling layers and three dense layers at the end, which transition from feature extraction to classification.

**MobileNetV3 Architecture**
The MobileNetV3 model is designed for mobile and edge device applications, emphasizing efficiency and performance. This model utilizes a combination of depthwise separable convolutions and squeeze-and-excitation modules to reduce computation while maintaining accuracy. The specific implementation modifies the classifier to adjust to the number of classes in the dataset.

### Training Components

We use **Adam optimizer** with a learning rate of 1e-5, chosen for its ability to handle sparse gradients and its efficiency in computation, and we employ **CrossEntropyLoss**, suitable for multi-class classification tasks.
A linear learning rate scheduler (**LinearLR**) adjusts the learning rate from an initial factor of 1.0 to an end factor of 0.3 over 40 iterations. This helps in gradually reducing the learning rate to fine-tune the model during training.
An instance of SaveModelWithBestValLoss is used to save the model with the **best validation loss**. This ensures that the best performing model on the validation set is preserved.


### Training loop

The training loop runs for up to 40 epochs, with each epoch consisting of a training and validation phase. During the training phase, the model is set to training mode. For each batch, inputs and labels are processed, and the loss is computed and backpropagated. The optimizer steps are taken, and the learning rate is updated. The training loss is accumulated and averaged at the end of each epoch.

During the validation phase, the model is set to evaluation mode. No gradient computations are performed. The validation loss is accumulated and averaged. If the validation loss does not improve over three consecutive epochs, early stopping is triggered to prevent overfitting.

## App/game <a name = "app"></a>



## Conclusions <a name = "conclusion"></a>


### Challenges
_(For example: challenges in gesture recognition include variability in lighting conditions, background noise, and differences in individual hand shapes and sizes. Robust algorithms and pre-processing steps are necessary to ensure accurate detection and interpretation.)_  

### Metrics and results
The performance of the models was evaluated using accuracy, precision, recall, and F1 score. Various configurations were tested, including different amounts of training data, use of data augmentation, and different model architectures.

| Model Configuration                                             | Accuracy      | Precision     | Recall        | F1 Score      |
|-----------------------------------------------------------------|---------------|---------------|---------------|---------------|
| MobileNetV3 with Data Augmentation (Full Dataset)               | 0.9952        | 0.9952        | 0.9952        | 0.9952        |
| MobileNetV3 with Data Augmentation (20% Train, 10% Test)        | 0.9754        | 0.9755        | 0.9754        | 0.9754        |
| MobileNetV3 without Data Augmentation (20% Train, 10% Test)     | 0.9699        | 0.9700        | 0.9699        | 0.9699        |
| ResNet50 with Data Augmentation (20% Train, 10% Test)           | 0.9864        | 0.9865        | 0.9864        | 0.9864        |
| ResNet50 without Data Augmentation (20% Train, 10% Test)           | 0.9861        | 0.9660        | 0.9861        | 0.9861        |
| Hand Segmentation (Frozen) + MobileNetV3 without Data Augmentation | 0.9657    | 0.9660        | 0.9657        | 0.9657        |
| Hand Segmentation (Unfrozen) + MobileNetV3 without Data Augmentation | 0.9478    | 0.9483        | 0.9478        | 0.9477        |
| Simple CNN Model without Data Augmentation (20% Train, 10% Test)| 0.6137        | 0.6122        | 0.6137        | 0.6103        |



### Discussion
The MobileNetV3 model, especially with data augmentation, demonstrated superior performance in terms of accuracy, precision, recall, and F1 score. ResNet50 also performed well, indicating its robustness. Simple CNN models showed significantly lower performance, highlighting the importance of advanced architectures and data augmentation in achieving high accuracy in classification tasks.

This detailed analysis provides insights into the effectiveness of different model architectures and training strategies, guiding future efforts in model selection and training methodologies for optimal performance.

## Demo <a name = "demo"></a>

## References <a name = "references"></a>
