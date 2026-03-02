🚗 Obstacle Avoidance Mobile Robot using CNN
📌 Project Overview

This project presents a vision-based obstacle avoidance mobile robot using a Convolutional Neural Network (CNN).
The robot captures real-time images using a camera, processes them through a trained CNN model, and predicts the direction to avoid obstacles.

The system enables autonomous navigation without human intervention.

🎯 Objectives

Develop a CNN model for obstacle detection

Enable autonomous navigation

Reduce collision risk

Implement real-time image processing

🧠 Technologies Used

Python 3.8

TensorFlow / Keras

OpenCV

NumPy

Raspberry Pi / Microcontroller

DC Motors & Motor Driver

⚙️ System Architecture

Camera → CNN Model → Direction Prediction → Motor Control → Robot Movement

🧪 Dataset

The dataset consists of labeled images categorized into:

Forward

Left

Right

Stop

Images were collected using the robot’s camera in different environments.

🏗️ CNN Model Architecture

Convolution Layer

ReLU Activation

MaxPooling

Flatten

Dense Layers

Softmax Output (4 classes)

📊 Model Performance

Accuracy: XX%

Loss: XX

Tested in real-time environment

🚀 How to Run
pip install -r requirements.txt
python train_model.py
python robot_control.py
📷 Output

The robot moves:

Forward when path is clear

Left or Right when obstacle detected

Stops when no safe path available

👩‍💻 Developed By

K. Kavya
Electronics and Communication Engineering