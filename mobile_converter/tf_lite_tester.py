# Script that uses the TFLite model to make predictions in order to test the conversion process

# Importing Libraries
import numpy as np
import cv2
import tensorflow as tf
from sklearn.preprocessing import LabelEncoder
import pickle

# Define the model folder
MODEL_FOLDER = "./mobile_converter"

# Load the TFLite model and allocate tensors.
interpreter = tf.lite.Interpreter(model_path=MODEL_FOLDER + "/model.tflite")
interpreter.allocate_tensors()

# Get input and output tensors.
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

# Load the label encoder
with open(MODEL_FOLDER + "/labelencoder.pkl", "rb") as le_dump_file:
    label_encoder: LabelEncoder = pickle.load(le_dump_file)

# Load the image and format to be used as a sample input
IMAGE_PATH = "D:\\Documents\\GitHub\\cv-gesture-recognition\\data\\classification\\test\\call\\0a8c60fc-9ef4-4de5-b601-3dee17ee110d.jpeg"
image = cv2.imread(IMAGE_PATH)
if image is None:
    raise FileNotFoundError(f"Image not found at path: {IMAGE_PATH}")
image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
image = cv2.resize(image, (300, 300))  # Resize to 300x300

# Reshape to (1, 3, 300, 300), from [1, 300, 300, 3]
image = np.expand_dims(image, axis=0)
image = np.transpose(image, (0, 3, 1, 2))  # Change to (1, 3, 300, 300)
image = image / 255.0  # Normalize the image

# Float32 is required for the model
image = image.astype(np.float32)

# Print the input shape
print(input_details[0]["shape"])

# Print the image shape
print(image.shape)


# Test the model on random input data
input_shape = input_details[0]["shape"]
interpreter.set_tensor(input_details[0]["index"], image)

interpreter.invoke()

# The function `get_tensor()` returns a copy of the tensor data.
# Use `tensor()` in order to get a pointer to the tensor.
output_data = interpreter.get_tensor(output_details[0]["index"])
print(output_data)

# Print the predicted label, given that output_data has the format: [[ 12.682318    1.6295592  -8.431072   -6.757467    8.70638    -0.5617069
#     4.5850606  -2.374082   -7.8380594   5.2861996 -11.046944   -1.3513894
#    -8.399297    0.8212291  -4.4889426  -6.731543    3.3007185 -13.5613365
#    -6.704102 ]]
# The label is the index of the maximum value
predicted_label = np.argmax(output_data)
print(label_encoder.inverse_transform([predicted_label]))

