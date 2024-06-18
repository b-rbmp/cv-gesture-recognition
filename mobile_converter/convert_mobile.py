# Script to convert the PyTorch model to TFLite format using Google's AI Edge Torch library

# PS: Only works on Linux

# Importing Libraries
import torch
import torch.nn as nn
import cv2
from sklearn.preprocessing import LabelEncoder
import pickle
from torchvision import models
import ai_edge_torch

# Define the model folder 
MODEL_FOLDER = "./mobile_converter"
MODEL_TYPE = "mobilenetv3" # Choose between "resnet50" and "mobilenetv3"

# Use GPU if available
torch.cuda.empty_cache()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)

with open(MODEL_FOLDER + "/labelencoder.pkl", "rb") as le_dump_file:
    label_encoder: LabelEncoder = pickle.load(le_dump_file)

# Generate the Model based on the model type
def generate_model(num_classes: int):
  if MODEL_TYPE == "resnet50":
    model = models.resnet50(pretrained=True)
    model.fc = nn.Linear(model.fc.in_features, num_classes)
  elif MODEL_TYPE == "mobilenetv3":
    model=models.mobilenet_v3_large(pretrained=True)
    num_features=model.classifier[0].in_features
    model.classifier=nn.Sequential(
        nn.Linear(in_features=num_features, out_features=4096, bias=True),
        nn.ReLU(inplace=True),
        nn.Dropout(p=0.5, inplace=False),
        nn.Linear(in_features=4096, out_features=4096, bias=True),
        nn.ReLU(inplace=True),
        nn.Dropout(p=0.5, inplace=False),
        nn.Linear(in_features=4096, out_features=num_classes, bias=True)
      )
    
  return model


# Build the model and load the weights
saved_model = generate_model(len(label_encoder.classes_))
saved_model.to(device)

# Load the best model
best_model = torch.load(MODEL_FOLDER+'/best_model.pth', map_location=device)

# Load the model weights
saved_model.load_state_dict(best_model['model_state_dict'])
saved_model.eval()

# Load the image and format to be used as a sample input
IMAGE_PATH = "/home/b-rbmp-ideapad/Downloads/cv/mobile/00ed026d-ee9d-46a3-b093-dbd05dc96f80.jpeg"
image = cv2.imread(IMAGE_PATH)
if image is None:
    raise FileNotFoundError(f"Image not found at path: {IMAGE_PATH}")
image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
image = cv2.resize(image, (300, 300))  # Resize to 300x300
image = (torch.tensor(image, dtype=torch.float32).permute(2, 0, 1) / 255.0).to(device)
image = image.unsqueeze(0)  # Add batch dimension

# Convert the model to tflite
edge_model = ai_edge_torch.convert(module=saved_model, sample_args=(image,))
edge_model.export("model.tflite")