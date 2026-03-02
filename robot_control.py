import cv2
import numpy as np
import tensorflow as tf
import RPi.GPIO as GPIO
import time

# Load trained model
model = tf.keras.models.load_model("model/trained_model.h5")

# Image size
IMG_HEIGHT = 64
IMG_WIDTH = 64

# GPIO Setup
GPIO.setmode(GPIO.BCM)

# Motor Pins (change if needed)
IN1 = 17
IN2 = 18
IN3 = 22
IN4 = 23

GPIO.setup(IN1, GPIO.OUT)
GPIO.setup(IN2, GPIO.OUT)
GPIO.setup(IN3, GPIO.OUT)
GPIO.setup(IN4, GPIO.OUT)

# Motor Control Functions
def forward():
    GPIO.output(IN1, True)
    GPIO.output(IN2, False)
    GPIO.output(IN3, True)
    GPIO.output(IN4, False)

def left():
    GPIO.output(IN1, False)
    GPIO.output(IN2, True)
    GPIO.output(IN3, True)
    GPIO.output(IN4, False)

def right():
    GPIO.output(IN1, True)
    GPIO.output(IN2, False)
    GPIO.output(IN3, False)
    GPIO.output(IN4, True)

def stop():
    GPIO.output(IN1, False)
    GPIO.output(IN2, False)
    GPIO.output(IN3, False)
    GPIO.output(IN4, False)

# Open Camera
cap = cv2.VideoCapture(0)

class_names = ['forward', 'left', 'right', 'stop']

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Resize image
    img = cv2.resize(frame, (IMG_WIDTH, IMG_HEIGHT))
    img = img / 255.0
    img = np.reshape(img, (1, IMG_HEIGHT, IMG_WIDTH, 3))

    # Predict
    prediction = model.predict(img)
    class_index = np.argmax(prediction)
    label = class_names[class_index]

    print("Prediction:", label)

    # Control Robot
    if label == 'forward':
        forward()
    elif label == 'left':
        left()
    elif label == 'right':
        right()
    else:
        stop()

    # Show camera
    cv2.imshow("Robot Camera", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
GPIO.cleanup()