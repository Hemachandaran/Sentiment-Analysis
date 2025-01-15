# train_model.py
import torch
import torch.optim as optim
import pandas as pd
from sentiment_model import SentimentModel

# Load your dataset (replace 'data.csv' with your actual data file)
data = pd.read_csv('data.csv')
X_train_reduced = data.iloc[:, :-1].values  # Features
y_train = data.iloc[:, -1].values            # Labels

# Convert training data to PyTorch tensors
X_train_tensor = torch.tensor(X_train_reduced, dtype=torch.float32)  
y_train_tensor = torch.tensor(y_train, dtype=torch.float32).view(-1, 1)

# Instantiate the model
input_dim = X_train_tensor.shape[1]
model = SentimentModel(input_dim)

# Define loss function and optimizer
criterion = nn.BCELoss()
optimizer = optim.Adam(model.parameters(), lr=0.012)

# Training loop (simplified for demonstration)
num_epochs = 10
for epoch in range(num_epochs):
    model.train()
    optimizer.zero_grad()
    outputs = model(X_train_tensor)
    loss = criterion(outputs, y_train_tensor)
    loss.backward()
    optimizer.step()
    print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}')

# Save the trained model's state dictionary
torch.save(model.state_dict(), 'sentiment_model.pth')


