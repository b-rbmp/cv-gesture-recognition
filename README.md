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

### Overview of the Project
Gesture-controlled robots represent a fascinating intersection of robotics and human-computer interaction, leveraging the natural human ability to communicate through gestures. This project aims to develop a basic gesture-controlled robot that utilizes motion sensors to interpret hand movements and control the robot accordingly. The robot can be used for various applications, such as interactive games or simple tasks, making it a versatile tool in both educational and recreational contexts.

### Importance and Applications of Gesture-Controlled Robots
Gesture-controlled robots have a wide range of applications. In educational settings, they can be used to teach students about robotics and programming in an interactive manner. In gaming, they provide a more immersive and intuitive way to control characters or objects. Additionally, in scenarios requiring hands-free control, such as in assistive technologies for individuals with disabilities, gesture-controlled robots offer significant advantages.

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

#### Capture Video Feed
Gesture recognition technology involves interpreting human gestures via mathematical algorithms. These gestures can originate from any bodily motion but are commonly focused on hand movements.
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
