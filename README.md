# Real-Time Gesture Recognition
Project for the Computer Vision 2023-2024 course at Sapienza

## Team

| **Name / Surname** | **Linkedin** | **GitHub** |
| :---: | :---: | :---: |
| `Bernardo Perrone De Menezes Bulcao Ribeiro ` | [![name](https://github.com/b-rbmp/NexxGate/blob/main/docs/logos/linkedin.png)](https://www.linkedin.com/in/b-rbmp/) | [![name](https://github.com/b-rbmp/NexxGate/blob/main/docs/logos/github.png)](https://github.com/b-rbmp) |
| `Roberta Chissich ` | [![name](https://github.com/b-rbmp/NexxGate/blob/main/docs/logos/linkedin.png)](https://www.linkedin.com/in/roberta-chissich/) | [![name](https://github.com/b-rbmp/NexxGate/blob/main/docs/logos/github.png)](https://github.com/RobCTs) |

## Overview
Hand Gesture Recognition is a technology that enables computers to interpret human gestures as commands, with applications ranging from robotics and gaming to assistive technologies and human-computer interaction. This project focuses on creating a real-time gesture recognition system to play Rock-Paper-Scissors against a computer.

## Table of Contents
+ [Introduction](#intro)
+ [Project Description](#project)
+ [System Architecture](#architecture)
+ [Implementation](#implementation)
+ [Application](#app)
+ [Evaluation and Results](#evaluation)
+ [Discussion](#aconclusion)
+ [Demo](#demo)
+ [References](#references)


## Introduction <a name = "intro"></a>

Hand Gesture Recognition is a technology that enables computers to interpret human gestures as commands. It works by capturing and analyzing the movements of the hands to recognize specific gestures and translate them into actions. This technology has a wide range of applications, from robotics and gaming to assistive technologies and human-computer interaction. Static Hand Gesture Recognition is a subset of this technology that focuses on recognizing static hand poses or shapes, which can be used to classify different gestures and commands based on a frame-by-frame analysis of the hand's position and shape.

### Importance and Applications 

One of the main applications of Hand Gesture Recognition is Human-Robot Interaction and Human Computer Interaction. By using hand gestures as commands, users can control robots, computers, and other devices in a more intuitive and natural way. This technology is particularly useful in scenarios where traditional input devices like keyboards and mice are not practical or feasible, such as in robotics, virtual reality, and augmented reality applications. Hand Gesture Recognition can also be used in healthcare, gaming, and security applications, among others.

## Project Description <a name = "project"></a>

### Objective
The primary objective of this project is to develop a real-time gesture recognition system that allows users to play a game of Rock-Paper-Scissors against a computer. The system captures live video feeds of the user's hand gestures using the camera, analyzes the movements to recognize the gestures using Deep Convolutional Neural Networks framed as a classification task, and translate them into commands to play the game. The computer will generate its own gestures randomly, and the system will determine the winner based on the rules of the game.

### Key Goals
**Motion Detection**: Utilize a standard smartphone/laptop camera to capture hand movements and translate them into commands.  
**Deep CNN Classification**: Implement a Deep Convolutional Neural Network (CNN) to classify hand gestures in real-time.
**Comparison of Models**: Compare the performance of different CNN models for gesture recognition, as well as the impact of data augmentation techniques.
**Mobile Application**: Implement a simple and intuitive mobile application that allows users to play the game using their smartphones.


## System Architecture <a name = "architecture"></a>

The system architecture for our hand gesture recognition project is designed to efficiently capture, process, and classify hand gestures in real-time. At a high level, the system comprises a data acquisition module, a pre-processing unit, a gesture recognition module powered by deep learning models, and an output module that interprets the recognized gestures into game commands for playing Rock-Paper-Scissors. The architecture leverages advanced neural network models and robust training methodologies to ensure high accuracy and responsiveness.

### Data Acquisition
The data acquisition module is responsible for capturing live video feeds of the user's hand gestures. This is typically done using a standard smartphone camera, which provides the necessary input data for further processing. The module ensures that the video feed is of sufficient quality and resolution to allow accurate gesture recognition.

### Pre-Processing Unit
Once the raw video data is captured, it undergoes a series of pre-processing steps. These steps include resizing the video frames, normalizing pixel values, and applying data augmentation techniques such as rotation, scaling, and flipping. Pre-processing is crucial for enhancing the robustness of the gesture recognition system by making the model invariant to various transformations and lighting conditions.

### Gesture Recognition Module
The core of the system is the gesture recognition module, which employs deep convolutional neural networks (CNNs) to classify hand gestures. We explored multiple model architectures for this purpose:

**VGG16**
The VGG16 model is a convolutional neural network (CNN) consisting of 13 convolutional layers followed by 3 fully connected layers. This architecture is known for its simplicity and uniformity in the size of the convolution filters. The network has five max-pooling layers and three dense layers at the end, which transition from feature extraction to classification.

**MobileNetV3**
The MobileNetV3 model is designed for mobile and edge device applications, emphasizing efficiency and performance. This model utilizes a combination of depthwise separable convolutions and squeeze-and-excitation modules to reduce computation while maintaining accuracy. The specific implementation modifies the classifier to adjust to the number of classes in the dataset.

**ResNet50**
The ResNet50 model, part of the Residual Networks (ResNet) family, is particularly well-suited for deep learning tasks due to its unique architecture that includes residual blocks. These blocks allow the model to learn identity mappings, which help in training very deep networks by mitigating the vanishing gradient problem. ResNet50 has 50 layers, making it a robust choice for capturing complex patterns in gesture recognition. Its performance is further enhanced by data augmentation techniques, leading to high accuracy and generalization.

The models are trianed using **HaGRID (HAnd Gesture Recognition Image Dataset)**. HaGRID is a comprehensive dataset containing 554,800 FullHD RGB images divided into 18 gesture classes and an additional no_gesture class for images with a second free hand. The dataset includes 723GB of data, with 410,800 images for training, 54,000 for validation, and 90,000 for testing, involving 37,583 unique individuals in various indoor lighting conditions. The subjects' ages range from 18 to 65, and the images were taken at distances of 0.5 to 4 meters from the camera, including extreme conditions like facing and backing to a window. 

[The HaGRID library on GitHub](https://github.com/hukenovs/hagrid)

### Training Components

We use **Adam optimizer** with a learning rate of 1e-5, chosen for its ability to handle sparse gradients and its efficiency in computation, and we employ **CrossEntropyLoss**, suitable for multi-class classification tasks.
A linear learning rate scheduler (**LinearLR**) adjusts the learning rate from an initial factor of 1.0 to an end factor of 0.3 over 40 iterations. This helps in gradually reducing the learning rate to fine-tune the model during training.
An instance of SaveModelWithBestValLoss is used to save the model with the **best validation loss**. This ensures that the best performing model on the validation set is preserved.


### Training loop

The training loop runs for up to 40 epochs, with each epoch consisting of a training and validation phase. During the training phase, the model is set to training mode. For each batch, inputs and labels are processed, and the loss is computed and backpropagated. The optimizer steps are taken, and the learning rate is updated. The training loss is accumulated and averaged at the end of each epoch.

During the validation phase, the model is set to evaluation mode. No gradient computations are performed. The validation loss is accumulated and averaged. If the validation loss does not improve over three consecutive epochs, early stopping is triggered to prevent overfitting.

### Output Module
The output module interprets the recognized gestures and translates them into commands for the Rock-Paper-Scissors game. The system determines the winner based on the recognized gestures of the user and the randomly generated gestures of the computer, providing real-time feedback to the user.

## Application <a name = "app"></a>



## Evaluation and Results <a name = "evaluation"></a>

### Comparison of All Models Without Augmented Data
The performance of the models without data augmentation was evaluated using accuracy, precision, recall, and F1 score. Then nalyzed to understand the baseline capabilities of each architecture. The results show a notable difference in performance:

| Model Configuration                                             | Accuracy      | Precision     | Recall        | F1 Score      |
|-----------------------------------------------------------------|---------------|---------------|---------------|---------------|
| MobileNetV3 without Data Augmentation (20% Train, 10% Test)     | 0.9699        | 0.9700        | 0.9699        | 0.9699        |
| ResNet50 without Data Augmentation (20% Train, 10% Test)           | 0.9846        | 0.9847        | 0.9846        | 0.9846        |
| Hand Segmentation (Frozen) + MobileNetV3 without Data Augmentation | 0.9657    | 0.9660        | 0.9657        | 0.9657        |
| Hand Segmentation (Unfrozen) + MobileNetV3 without Data Augmentation | 0.9478    | 0.9483        | 0.9478        | 0.9477        |
| Simple CNN Model without Data Augmentation (20% Train, 10% Test)| 0.6137        | 0.6122        | 0.6137        | 0.6103        |

The comparison highlights the significant impact of advanced model architectures like ResNet50 and MobileNetV3 over simpler CNN models. Even without augmentation, these advanced models achieve high accuracy, demonstrating their robustness in classification tasks.

### ResNet Model Comparison with and without Data Augmentation

To further understand the impact of data augmentation, we compared the ResNet50 model's performance with and without augmented data:

| Model Configuration                                             | Accuracy      | Precision     | Recall        | F1 Score      |
|-----------------------------------------------------------------|---------------|---------------|---------------|---------------|
| ResNet50 with Data Augmentation (20% Train, 10% Test)           | 0.9864        | 0.9865        | 0.9864        | 0.9864        |
| ResNet50 without Data Augmentation (20% Train, 10% Test)           | 0.9846        | 0.9847        | 0.9846        | 0.9846        |

The results indicate that while the ResNet50 model performs exceptionally well without data augmentation, applying data augmentation slightly enhances its precision, ensuring a more balanced and generalized performance. The marginal improvement suggests that ResNet50 is highly effective in recognizing gestures even with a limited amount of training data, but data augmentation can still provide an edge in terms of precision.

### Challenges
_(For example: challenges in gesture recognition include variability in lighting conditions, background noise, and differences in individual hand shapes and sizes. Robust algorithms and pre-processing steps are necessary to ensure accurate detection and interpretation.)_  

### Discussion
The performance analysis of various models in this project highlights the significant impact of advanced architectures and data augmentation on the accuracy and robustness of hand gesture recognition systems.

The ResNet50 model demonstrates high performance both with and without data augmentation. When trained with data augmentation on 20% of the training set and tested on 10% of the test set, ResNet50 achieved an accuracy of 98.64%, precision of 98.65%, recall of 98.64%, and an F1 score of 98.64%. Without data augmentation, the ResNet50 model showed a marginally lower accuracy of 98.61%, precision of 96.60%, recall of 98.61%, and an F1 score of 98.61%. The slight improvement in precision with data augmentation suggests enhanced model robustness and generalization, making it better suited for real-world variations in hand gestures.

The comparative analysis of models without data augmentation highlights the superior performance of the ResNet50 architecture. ResNet50 without data augmentation achieved an accuracy of 98.61%, precision of 96.60%, recall of 98.61%, and an F1 score of 98.61%. In comparison, the hand segmentation model with a frozen MobileNetV3 backbone reached an accuracy of 96.57%, precision of 96.60%, recall of 96.57%, and an F1 score of 96.57%. When the hand segmentation model was unfrozen, it achieved an accuracy of 94.78%, precision of 94.83%, recall of 94.78%, and an F1 score of 94.77%. The simple CNN model showed significantly lower performance, with an accuracy of 61.37%, precision of 61.22%, recall of 61.37%, and an F1 score of 61.03%.

Qualitative observations further reinforce the quantitative findings. During training, ResNet50 exhibited greater stability, with consistent convergence and minimal fluctuations in loss and accuracy. The model also demonstrated superior generalization, handling variations in lighting, background, and hand positioning more effectively, particularly when data augmentation was applied. Although advanced models like ResNet50 required more sophisticated implementation and computational resources, the trade-off was justified by the substantial gains in performance and robustness.

In summary, this detailed analysis underscores the effectiveness of the ResNet50 architecture in achieving high accuracy and robustness in hand gesture recognition tasks. The use of data augmentation further enhances model performance, particularly in terms of precision. These findings provide valuable insights for future model selection and training strategies, emphasizing the importance of advanced architectures and data augmentation in developing reliable and accurate gesture recognition systems.

## Demo <a name = "demo"></a>

## References <a name = "references"></a>
