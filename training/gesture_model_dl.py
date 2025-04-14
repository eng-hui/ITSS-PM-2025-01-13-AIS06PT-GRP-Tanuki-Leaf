import json
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split

# 1. Load and prepare the data from gesture_library.json
class HandGestureDataset(Dataset):
    def __init__(self, features, labels):
        # Reshape features to make them compatible with CNN
        # Reshape to [batch_size, 1, 21, 3] - 1 channel, 21 joints, 3 coordinates per joint
        self.features = torch.tensor(features, dtype=torch.float32).view(-1, 1, 21, 3)
        self.labels = torch.tensor(labels, dtype=torch.long)
    
    def __len__(self):
        return len(self.features)
    
    def __getitem__(self, idx):
        return self.features[idx], self.labels[idx]

def load_gesture_data(json_file):
    with open(json_file, 'r') as f:
        gesture_lib = json.load(f)
    
    features = []
    labels = []
    label_map = {}
    
    # Map gesture names to numeric labels
    for idx, gesture_name in enumerate(gesture_lib.keys()):
        label_map[gesture_name] = idx
    
    # Convert data to features and labels
    for gesture_name, instances in gesture_lib.items():
        for instance in instances:
            # Organize the data in a matrix form (21 joints × 3 coordinates)
            hand_matrix = np.zeros((21, 3))
            
            for i, landmark in enumerate(instance[:21]):  # Make sure we only use 21 landmarks
                try:
                    # Handle dictionary format (with .get method)
                    if isinstance(landmark, dict):
                        hand_matrix[i, 0] = landmark.get('x', 0)
                        hand_matrix[i, 1] = landmark.get('y', 0)
                        hand_matrix[i, 2] = landmark.get('z', 0)
                    # Handle string format
                    elif isinstance(landmark, str):
                        # Try to parse as JSON if it's a string representation of a dictionary
                        try:
                            landmark_dict = json.loads(landmark)
                            hand_matrix[i, 0] = landmark_dict.get('x', 0)
                            hand_matrix[i, 1] = landmark_dict.get('y', 0)
                            hand_matrix[i, 2] = landmark_dict.get('z', 0)
                        except json.JSONDecodeError:
                            # If it can't be parsed as JSON, just set zeros
                            hand_matrix[i, :] = 0
                    # Handle list/array format
                    elif isinstance(landmark, list):
                        for j, coord in enumerate(landmark[:3]):  # Take only x,y,z
                            hand_matrix[i, j] = float(coord)
                    # Handle any other format
                    else:
                        hand_matrix[i, :] = 0
                except Exception as e:
                    print(f"Error processing landmark {i} for gesture {gesture_name}: {e}")
                    print(f"Landmark value: {landmark}")
                    hand_matrix[i, :] = 0
            
            features.append(hand_matrix.flatten())
            labels.append(label_map[gesture_name])
    
    return np.array(features), np.array(labels), label_map

# 2. Define the CNN model
class HandGestureCNN(nn.Module):
    def __init__(self, num_classes):
        super(HandGestureCNN, self).__init__()
        
        # Convolutional layers - modified to handle the small width dimension
        # First convolution - preserve width dimension by using kernel_size=(3,1)
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=32, kernel_size=(3,1), padding=(1,0))
        self.bn1 = nn.BatchNorm2d(32)
        
        # Second convolution - preserve width dimension
        self.conv2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=(3,1), padding=(1,0))
        self.bn2 = nn.BatchNorm2d(64)
        
        # Third convolution - preserve width dimension
        self.conv3 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=(3,1), padding=(1,0))
        self.bn3 = nn.BatchNorm2d(128)
        
        # Activation and pooling - only pool along height dimension
        self.relu = nn.ReLU()
        self.pool = nn.MaxPool2d(kernel_size=(2,1), stride=(2,1))
        
        # Calculate the size of flattened features after convolution and pooling
        self.fc_input_size = self._get_conv_output_size()
        
        # Fully connected layers
        self.fc1 = nn.Linear(self.fc_input_size, 128)
        self.dropout = nn.Dropout(0.3)
        self.fc2 = nn.Linear(128, num_classes)
    
    def _get_conv_output_size(self):
        # Dummy forward pass to calculate the size of flattened features
        x = torch.zeros(1, 1, 21, 3)
        x = self.pool(self.relu(self.bn1(self.conv1(x))))  # Height dimension halved
        x = self.pool(self.relu(self.bn2(self.conv2(x))))  # Height dimension halved again
        x = self.relu(self.bn3(self.conv3(x)))
        return x.numel()  # Get total number of elements
    
    def forward(self, x):
        # Convolutional layers with batch normalization and activation
        x = self.pool(self.relu(self.bn1(self.conv1(x))))
        x = self.pool(self.relu(self.bn2(self.conv2(x))))
        x = self.relu(self.bn3(self.conv3(x)))
        
        # Flatten the output from convolutional layers
        x = x.view(x.size(0), -1)
        
        # Fully connected layers
        x = self.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        return x

# 3. Training function
def train_model(model, train_loader, val_loader, criterion, optimizer, num_epochs=50):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    
    best_val_acc = 0.0
    
    for epoch in range(num_epochs):
        # Training phase
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0
        
        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)
             
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item() * inputs.size(0)
            
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
        
        epoch_loss = running_loss / len(train_loader.dataset)
        epoch_acc = correct / total
        
        # Validation phase
        model.eval()
        val_loss = 0.0
        val_correct = 0
        val_total = 0
        
        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                
                val_loss += loss.item() * inputs.size(0)
                
                _, predicted = torch.max(outputs, 1)
                val_total += labels.size(0)
                val_correct += (predicted == labels).sum().item()
        
        val_epoch_loss = val_loss / len(val_loader.dataset)
        val_epoch_acc = val_correct / val_total
        
        print(f'Epoch {epoch+1}/{num_epochs}:')
        print(f'Train Loss: {epoch_loss:.4f}, Train Acc: {epoch_acc:.4f}')
        print(f'Val Loss: {val_epoch_loss:.4f}, Val Acc: {val_epoch_acc:.4f}')
        
        # Save best model
        if val_epoch_acc > best_val_acc:
            best_val_acc = val_epoch_acc
            torch.save(model.state_dict(), 'best_hand_gesture_cnn_model.pth')
            print(f'Model saved with validation accuracy: {best_val_acc:.4f}')
    
    return model

# 4. Main execution
def main():
    # Load data
    features, labels, label_map = load_gesture_data('gesture_library.json')
    
    # Split data into training and validation sets
    X_train, X_val, y_train, y_val = train_test_split(features, labels, test_size=0.2, random_state=42)
    
    # Create datasets and data loaders
    train_dataset = HandGestureDataset(X_train, y_train)
    val_dataset = HandGestureDataset(X_val, y_val)
    
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=32)
    
    # Initialize model, loss function, and optimizer
    num_classes = len(label_map)
    
    model = HandGestureCNN(num_classes)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-4)
    
    # Train the model
    trained_model = train_model(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        criterion=criterion,
        optimizer=optimizer,
        num_epochs=50
    )
    
    # Save the final model
    torch.save({
        'model_state_dict': trained_model.state_dict(),
        'label_map': label_map,
        'num_classes': num_classes
    }, 'hand_gesture_cnn_model_full.pth')
    
    print("Training completed and CNN model saved!")

if __name__ == "__main__":
    main()

def load_and_predict(model_path, input_data):
    # Load the saved model
    checkpoint = torch.load(model_path)
    
    # Recreate the model architecture
    model = HandGestureCNN(num_classes=checkpoint['num_classes'])
    
    # Load the trained weights
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    # Reshape input data for CNN (1, 1, 21, 3)
    input_data = np.array(input_data).reshape(1, 1, 21, 3)
    input_tensor = torch.tensor(input_data, dtype=torch.float32)
    
    # Make prediction
    with torch.no_grad():
        output = model(input_tensor)
        _, predicted = torch.max(output, 1)
        
    # Map the predicted index back to gesture name
    label_map = checkpoint['label_map']
    reverse_label_map = {v: k for k, v in label_map.items()}
    
    predicted_gesture = reverse_label_map[predicted.item()]
    return predicted_gesture