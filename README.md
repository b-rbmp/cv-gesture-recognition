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
+ [Design](#design)
+ [Conclusions](#conclusion)
+ [Demo](#demo)
+ [References](#references)


## Introduction <a name = "intro"></a>

Gesture-controlled robots represent a fascinating intersection of robotics and human-computer interaction, leveraging the natural human ability to communicate through gestures. This project aims to develop a basic gesture-controlled robot that utilizes motion sensors to interpret hand movements and control the robot accordingly. Gesture-controlled robots are designed to function based on signals given through hand gestures, employing technologies such as accelerometers or image processing to detect and interpret these gestures​ (Roboversity, 2024; DelPreto & Rus, 2020). The robot can be used for various applications, such as interactive games or simple tasks, making it a versatile tool in both educational and recreational contexts.

### Importance and Applications of Gesture-Controlled Robots
Gesture-controlled robots have a wide range of applications. In educational settings, they can be used to teach students about robotics and programming interactively, making learning more engaging and hands-on. By using gesture-based controls, students can see immediate, tangible results from their commands, fostering a deeper understanding of the principles of robotics and programming​​ (Roboversity, 2024).

In gaming, gesture-controlled robots provide a more immersive and intuitive way to control characters or objects. This can lead to more interactive and engaging gaming experiences, where players use natural movements to interact with the game environment. Moreover, such technology can be used to develop new forms of interactive entertainment and educational tools that combine physical activity with digital content.

Additionally, in scenarios requiring hands-free control, gesture-controlled robots offer significant advantages. For instance, in assistive technologies for individuals with disabilities, these robots can enable users to perform tasks that would otherwise be difficult or impossible. By interpreting simple hand gestures, these robots can assist with daily activities, providing greater independence and improving the quality of life for users.

By leveraging the power of gesture recognition and human-computer interaction, gesture-controlled robots open up new possibilities across various fields, from education and entertainment to healthcare and assistive technology. These advancements underscore the potential of robotics to enhance human capabilities and interaction with technology.

## Project Description <a name = "project"></a>

### Objective
The primary objective of this project is to design and implement a gesture-controlled robot using a webcam and motion detection software. The system should be capable of recognizing basic hand gestures to control the robot's movements, providing an intuitive interface for the user.

### Key Features
**Motion Detection**: Utilize a standard webcam to capture hand movements and translate them into commands.  
**Control Modes**: The GUI allows users to switch between different modes of operation, such as follow mode and avoid mode.  
**User-Friendly Interface**: Develop a simple and intuitive GUI for controlling the robot and switching between modes.  

## System components <a name = "architecture"></a>

### Hardware
**Webcam**
The system employs a standard webcam to capture live video feeds of the user's hand gestures. Webcams are chosen for their accessibility and ease of integration with computer systems.

**Robot**
The robot used in this project is a simple, programmable mobile simulation platform for Martino robotos, equipped with basic movement capabilities such as moving forward, backward, and turning.

### Software
**Motion Detection Software**
The core of the gesture control system is the motion detection software. This software processes the video feed from the webcam to identify and interpret hand gestures. _(talk about techniques for motion detection such as background subtraction, frame differencing, and optical flow)_

**GUI**
The graphical user interface (GUI) provides the user with an easy-to-use platform to control the robot. It displays the live video feed from the webcam and offers controls to switch between different operational modes._(say something more once we implement it)_

## Design <a name = "design"></a>
The system architecture comprises three main components: the input module (webcam), the processing module (motion detection software), and the output module (robot). The GUI acts as an intermediary, providing user interaction capabilities.

### Software Workflow
**Capture Video Feed**: The webcam captures the live video feed.  
**Process Video Feed**: The motion detection software analyzes the video feed to detect hand gestures.  
**Generate Commands**: Based on the detected gestures, commands are generated and sent to the robot.  

#### Gesture Recognition Technology
HaGRID - HAnd Gesture Recognition Image Dataset
We utilized the **HaGRID (HAnd Gesture Recognition Image Dataset)** to train our gesture recognition model. HaGRID is a comprehensive dataset containing 554,800 FullHD RGB images divided into 18 gesture classes and an additional no_gesture class for images with a second free hand. The dataset includes 723GB of data, with 410,800 images for training, 54,000 for validation, and 90,000 for testing, involving 37,583 unique individuals in various indoor lighting conditions. The subjects' ages range from 18 to 65, and the images were taken at distances of 0.5 to 4 meters from the camera, including extreme conditions like facing and backing to a window. 

[The HaGRID library on GitHub](https://github.com/hukenovs/hagrid)

In order to be able to recognize hand-gestures, we employed several techniques, including:

_(Background Subtraction: Identifying moving objects in a video feed by subtracting the background.
Frame Differencing: Detecting changes between consecutive video frames.
Optical Flow: Tracking the motion of objects between frames.)_

#### Process Video Feed

#### Generate Commands


## Conclusions <a name = "conclusion"></a>
### Challenges
_(For example: challenges in gesture recognition include variability in lighting conditions, background noise, and differences in individual hand shapes and sizes. Robust algorithms and pre-processing steps are necessary to ensure accurate detection and interpretation.)_  

### Metrics and results
_(Provide some numbers in order to be able to conclude something)_

### Discussion
_(Conclusions)_

## Demo <a name = "demo"></a>
## References <a name = "references"></a>
Joseph DelPreto and Daniela Rus. 2020. Plug-and-Play Gesture Control Using Muscle and Motion Sensors. In Proceedings of the 2020 ACM/IEEE International Conference on Human-Robot Interaction (HRI ’20), March 23–26, 2020, Cambridge, United Kingdom. ACM, New York, NY, USA, 10 pages. https://doi.org/10.1145/3319502.3374823
Roboversity. (n.d.). What is Gesture Based Robots? Retrieved from https://www.roboversity.com/resources/What-is-gesture-based-robots, access last 14/06/2024.
