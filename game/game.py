# Script that uses one of the trained models to play Rock Paper Scissors against the computer using the laptop's camera.
# The player has 3 seconds to show their move, and the computer will randomly choose a move.
# The game will last for 5 rounds, and the player will be asked if they want to play again at the end of the game.
# The game will display the player's move, the computer's move, and the result of each round.
# The final score and the game result will be displayed at the end of the game.

# Importing Libraries
import time
from sklearn.preprocessing import LabelEncoder
import torch
import torch.nn as nn
import torch.nn.functional as F
import cv2
from torchvision import models
import random

# Use GPU if available
torch.cuda.empty_cache()
device = torch.device("cpu")
print(f"Using device: {device}")

# Load the label encoder from pickle file
import pickle
MODEL_FOLDER = ".\\resnet50"
MODEL_TYPE = "resnet50" # Change to "mobilenetv3" to use MobileNetV3 Large model
with open(MODEL_FOLDER + "\\labelencoder.pkl", "rb") as le_dump_file:
    label_encoder: LabelEncoder = pickle.load(le_dump_file)

# Generate the model based on the model type
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

# Function that allows for one prediction to be made for a single image using the trained model
def predict_single_image(model, image, label_encoder):
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = cv2.resize(image, (300, 300))  # Resize to 300x300
    image = torch.tensor(image, dtype=torch.float32).permute(2, 0, 1) / 255.0
    image = image.unsqueeze(0)  # Add batch dimension

    # Make prediction
    model.eval()
    with torch.no_grad():
        logits = model(image.to(device))
        preds = torch.argmax(logits, dim=-1)

    # Decode the prediction
    label = label_encoder.inverse_transform([preds.item()])[0]
    return label

# build the model, without loading the pre-trained weights or fine-tune layers
saved_model = generate_model(len(label_encoder.classes_))
saved_model.to(device)
best_model = torch.load(MODEL_FOLDER + '\\best_model_FINAL.pth')
saved_model.load_state_dict(best_model['model_state_dict'])

# Use camera to continuously predict the image
import cv2

# Use camera to continuously predict the image
cap = cv2.VideoCapture(0)

# Rock Paper Scissors conversion to the dataset classes
rock = ["fist"]
paper = ["palm", "stop", "stop_inverted"]
scissors = ["peace", "peace_inverted"]

# Game Parameters
player_score = 0
current_round = 1
total_rounds = 5
DURATION_CAPTURE = 3

# Function to convert the prediction to rock, paper, or scissors
def convert_prediction(prediction):
    converted_prediction = None
    # Convert the player move to rock, paper, or scissors
    if prediction in rock:
        converted_prediction = "rock"
    elif prediction in paper:
        converted_prediction = "paper"
    elif prediction in scissors:
        converted_prediction = "scissors"
    else:
        converted_prediction = "Not Valid"
    return converted_prediction

# Function to capture the player's move. It works by capturing the player's move for a certain duration, generating predictions for each frame.
# The function will return a list of predictions for each frame.
def capture_move(duration=DURATION_CAPTURE):
    # Variables for capturing the player's move
    predictions = []
    start_time = time.time()
    end_time = start_time + duration
    last_prediction = None
    counter_last_prediction = 0
    frame_update_frequency_last_prediction = 5

    # Capture the player's move for a certain duration
    while time.time() < end_time:
        ret, frame = cap.read()
        if not ret:
            print("Error: failed to capture image")
            break

        # Keep the frame dimensions intact
        frame_height, frame_width = frame.shape[:2]

        # Create the biggest possible square in the center of the frame that will be cropped
        min_dim = min(frame_height, frame_width)
        start_x = frame_width // 2 - min_dim // 2
        end_x = frame_width // 2 + min_dim // 2
        start_y = frame_height // 2 - min_dim // 2
        end_y = frame_height // 2 + min_dim // 2
        cropped_frame = frame[start_y:end_y, start_x:end_x]

        prediction = predict_single_image(saved_model, cropped_frame, label_encoder)
        predictions.append(prediction)

        # Display capture status on frame with a black rectangle as background
        cv2.rectangle(frame, (10, 10), (200, 40), (0, 0, 0), -1)
        cv2.putText(frame, "State your move...", (15, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)

        if counter_last_prediction >= frame_update_frequency_last_prediction:
            last_prediction = convert_prediction(prediction)
            counter_last_prediction = 0
        else:
            counter_last_prediction += 1
        # Display the current prediction on a black rectangle as background
        if last_prediction:
            cv2.rectangle(frame, (frame_width - 210, 10), (frame_width - 10, 40), (0, 0, 0), -1)
            cv2.putText(frame, f"Prediction: {last_prediction}", (frame_width - 205, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
            
        cv2.imshow("frame", frame)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    return predictions

# Main loop for the game
while True:
    # Reset the game parameters every game of 5 rounds
    player_score = 0
    current_round = 1
    
    # Play one game of Rock Paper Scissors:
    for current_round in range(1, total_rounds + 1):
        # Generate a random move for the computer between rock, paper, and scissors
        computer_move = random.choice(["rock", "paper", "scissors"])

        # Capture the player's move for each frame using the camera and model
        predictions = capture_move(duration=DURATION_CAPTURE)
        
        # If the player's move was captured, determine the winner of the round
        if predictions:
            # Refresh the frame
            ret, frame = cap.read()
            if not ret:
                print("Error: failed to capture image")
                break

            # Get the frame dimensions
            frame_height, frame_width = frame.shape[:2]

            # Get the prediction with the highest frequency
            average_prediction = max(set(predictions), key=predictions.count)

            # Convert the player move to rock, paper, or scissors
            average_prediction = convert_prediction(average_prediction)

            # Determine the winner of the round
            if average_prediction == computer_move:
                result_text = "It's a tie!"
            elif (average_prediction == "rock" and computer_move == "scissors") or \
                 (average_prediction == "paper" and computer_move == "rock") or \
                 (average_prediction == "scissors" and computer_move == "paper"):
                result_text = "Player Wins!"
                player_score += 1
            else:
                result_text = "Computer Wins!"
                player_score -= 1

            # Add black rectangles as backgrounds for the text
            cv2.rectangle(frame, (10, 10), (300, 80), (0, 0, 0), -1)
            cv2.rectangle(frame, (frame_width - 310, 10), (frame_width - 10, 80), (0, 0, 0), -1)
            cv2.rectangle(frame, (10, 420), (frame_width - 10, 460), (0, 0, 0), -1)

            cv2.putText(frame, f"Round {current_round}/{total_rounds}", (20, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
            cv2.putText(frame, f"Player Score: {player_score}", (20, 65), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
            
            # Display the player move
            cv2.putText(frame, f"Player Move: {average_prediction}", (frame_width - 300, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
            # Display the computer move
            cv2.putText(frame, f"Computer Move: {computer_move}", (frame_width - 300, 65), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)

            # Display the round result
            cv2.putText(frame, result_text, ((frame_width - cv2.getTextSize(result_text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)[0][0]) // 2, 445), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)

            # Display the frame with the round results
            cv2.imshow("frame", frame)
            cv2.waitKey(3000)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break
        
    # Refresh the frame
    ret, frame = cap.read()
    if not ret:
        print("Error: failed to capture image")
        break

    frame_height, frame_width = frame.shape[:2]

    # Display the final score
    cv2.rectangle(frame, (10, 250), (frame_width - 10, 290), (0, 0, 0), -1)
    cv2.putText(frame, f"Final Score: {player_score}", (20, 270), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)

    # Display the game result
    if player_score > 0:
        game_result_text = "Player Wins the Game!"
    elif player_score < 0:
        game_result_text = "Computer Wins the Game!"
    else:
        game_result_text = "It's a tie!"
    
    cv2.rectangle(frame, (10, 290), (frame_width - 10, 330), (0, 0, 0), -1)
    cv2.putText(frame, game_result_text, (20, 310), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
    cv2.imshow("frame", frame)
    cv2.waitKey(3000)
    
    # Ask the player if they want to play again
    cv2.rectangle(frame, (10, 330), (frame_width - 10, 370), (0, 0, 0), -1)
    cv2.putText(frame, "Play again? (y/n)", (20, 350), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
    cv2.imshow("frame", frame)
    if cv2.waitKey(0) & 0xFF == ord("n"):
        break