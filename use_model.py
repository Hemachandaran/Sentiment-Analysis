# use_model.py
import torch
import pandas as pd
from sentiment_model import SentimentModel

# Load the trained model's state dictionary
input_dim = 10  # Set this to match your input dimension from training
model = SentimentModel(input_dim)

# Load the saved weights into the model
model.load_state_dict(torch.load('sentiment_model.pth'))
model.eval()  # Set the model to evaluation mode

# Prepare new data for prediction (replace with actual new data)
new_data_array = [[0.5] * input_dim]  # Example input; replace with your own data
new_data_tensor = torch.FloatTensor(new_data_array)

# Make predictions using the loaded model
with torch.no_grad():
    predictions = model(new_data_tensor)

predicted_class = (predictions > 0.5).int()  # Convert probabilities to binary class
print("Predicted class:", predicted_class.numpy())
