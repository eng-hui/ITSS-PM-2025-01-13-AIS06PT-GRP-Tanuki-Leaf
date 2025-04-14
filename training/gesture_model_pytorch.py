import json
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import LabelEncoder

# Load the gesture library JSON file
with open('gesture_library.json', 'r') as f:
    gesture_data = json.load(f)

# Extract data and labels
data = []
labels = []

for hand_side in gesture_data:
    for gesture_id in gesture_data[hand_side]:
        for gesture in gesture_data[hand_side][gesture_id]:
            data.append(gesture)
            # Use the string label directly
            labels.append(gesture_id)

# Convert to numpy arrays
data = np.array(data)
labels = np.array(labels)

print("Data shape:", data.shape)
print("Labels shape:", labels.shape)

# Convert string labels to integers using LabelEncoder
label_encoder = LabelEncoder()
labels_encoded = label_encoder.fit_transform(labels)

# Store label mapping for later reference
label_mapping = dict(zip(label_encoder.classes_, range(len(label_encoder.classes_))))
print("Label mapping:", label_mapping)

# Flatten the data
data = data.reshape(data.shape[0], -1)  # Flatten the data

# Shuffle the data
indices = np.arange(data.shape[0])
np.random.shuffle(indices)
data = data[indices]
labels_encoded = labels_encoded[indices]

# Split the data into training and testing sets
split_ratio = 0.8
split_index = int(split_ratio * data.shape[0])
X_train, X_test = data[:split_index], data[split_index:]
y_train, y_test = labels_encoded[:split_index], labels_encoded[split_index:]

# Convert to PyTorch tensors
X_train_tensor = torch.FloatTensor(X_train)
X_test_tensor = torch.FloatTensor(X_test)
y_train_tensor = torch.LongTensor(y_train)
y_test_tensor = torch.LongTensor(y_test)

# Create DataLoader - FIXED: Using tensor variables instead of numpy arrays
train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
test_dataset = TensorDataset(X_test_tensor, y_test_tensor)
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

# Define the model
class GestureModel(nn.Module):
    def __init__(self, input_size, num_classes):
        super(GestureModel, self).__init__()
        self.fc1 = nn.Linear(input_size, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, 32)
        self.fc4 = nn.Linear(32, num_classes)
    
    def forward(self, x):
        # Use F.relu instead of torch.relu to avoid potential name conflict
        x = torch.nn.functional.relu(self.fc1(x))
        x = torch.nn.functional.relu(self.fc2(x))
        x = torch.nn.functional.relu(self.fc3(x))
        x = self.fc4(x)
        return x

input_size = X_train.shape[1]
# Use label_encoder.classes_ to get the exact number of classes
num_classes = len(label_encoder.classes_)
model = GestureModel(input_size, num_classes)

# Define loss and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Training loop
num_epochs = 50
for epoch in range(num_epochs):
    model.train()
    for X_batch, y_batch in train_loader:
        optimizer.zero_grad()
        outputs = model(X_batch)
        loss = criterion(outputs, y_batch)
        loss.backward()
        optimizer.step()
    
    # Validation
    model.eval()
    with torch.no_grad():
        correct = 0
        total = 0
        for X_batch, y_batch in test_loader:
            outputs = model(X_batch)
            _, predicted = torch.max(outputs.data, 1)
            total += y_batch.size(0)
            correct += (predicted == y_batch).sum().item()
        accuracy = correct / total
    print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}, Accuracy: {accuracy:.4f}')

# Save the trained model
torch.save(model.state_dict(), 'hand_gesture_nn.pth')