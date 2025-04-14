import json
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score, classification_report
from sklearn.preprocessing import OneHotEncoder
import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier

# Load the JSON file
with open('gesture_library.json', 'r') as f:
    data_json = json.load(f)

# Choose a target length for each sample (adjust as needed)
target_length = 40

X = []
y = []
for hand in data_json:
    for gesture_id in data_json[hand]:
        for sample in data_json[hand][gesture_id]:
            # Flatten the sample (list of coordinate pairs) to a 1D vector
            flat_sample = np.array(sample).flatten()
            # Remove all zero values from this flattened sample
            filtered_sample = flat_sample[flat_sample != 0]
            # If after removing zeros the sample is empty, skip it:
            if filtered_sample.size == 0:
                continue
            # Pad or truncate the filtered sample to a fixed length:
            if filtered_sample.size < target_length:
                # Pad with zeros at the end
                pad_width = target_length - filtered_sample.size
                modified_sample = np.pad(filtered_sample, (0, pad_width), mode='constant')
            elif filtered_sample.size > target_length:
                # Truncate to the target length
                modified_sample = filtered_sample[:target_length]
            else:
                modified_sample = filtered_sample
            X.append(modified_sample)
            y.append(gesture_id)

X = np.array(X)
y = np.array(y)

print("Data shape:", X.shape)
print("Labels shape:", y.shape)

# One-hot encode the string labels
encoder = OneHotEncoder(sparse_output=False)  # Updated for sklearn 1.6.1
y_encoded = encoder.fit_transform(y.reshape(-1, 1))

# Split into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(
    X, y_encoded, test_size=0.2, random_state=42
)

# Create and train a classifier
model = MLPClassifier(hidden_layer_sizes=(100,), max_iter=1000, random_state=42)
model.fit(X_train, y_train)

# Print classes for MLP model
mlp_classes = encoder.categories_[0]
print("MLP Classes:", mlp_classes)

# Evaluate the model using predicted probabilities to avoid all-zero prediction errors
y_proba = model.predict_proba(X_test)
# Get the index with the highest probability for each sample
pred_indices = np.argmax(y_proba, axis=1)
# Convert indices to labels using the encoder categories
y_pred = np.array([encoder.categories_[0][i] for i in pred_indices])
y_true = encoder.inverse_transform(y_test)

print("Accuracy:", accuracy_score(y_true, y_pred))
print("\nMLP Classification Report:\n", classification_report(y_true, y_pred))
joblib.dump(model, 'hand_gesture_mlp.joblib')  # Save to disk

# Random Forest Classifier
label_encoder = LabelEncoder()
y_labels = label_encoder.fit_transform(y)
X_train_rf, X_test_rf, y_train_rf, y_test_rf = train_test_split(X, y_labels, test_size=0.2, random_state=42)
model_r = RandomForestClassifier(n_estimators=100, random_state=42)
model_r.fit(X_train_rf, y_train_rf)

# Print classes for Random Forest model
rf_classes = label_encoder.classes_
print("Random Forest Classes:", rf_classes)

y_pred_r = model_r.predict(X_test_rf)

print("Random Forest Accuracy:", accuracy_score(y_test_rf, y_pred_r))
print("\nRandom Forest Classification Report:\n", classification_report(
    y_test_rf, y_pred_r, target_names=label_encoder.classes_, labels=np.arange(len(label_encoder.classes_))
))
joblib.dump(model_r, 'hand_gesture_rf.joblib')  # Save to disk

# Decision Tree Classifier
X_train_dt, X_test_dt, y_train_dt, y_test_dt = train_test_split(X, y_labels, test_size=0.2, random_state=42)
model_dt = DecisionTreeClassifier(random_state=42)
model_dt.fit(X_train_dt, y_train_dt)

# Print classes for Decision Tree model
dt_classes = label_encoder.classes_
print("Decision Tree Classes:", dt_classes)

y_pred_dt = model_dt.predict(X_test_dt)

print("Decision Tree Accuracy:", accuracy_score(y_test_dt, y_pred_dt))
print("\nDecision Tree Classification Report:\n", classification_report(
    y_test_dt, y_pred_dt, target_names=label_encoder.classes_, labels=np.arange(len(label_encoder.classes_))
))
joblib.dump(model_dt, 'hand_gesture_dt.joblib')  # Save to disk

# SVM Classifier
X_train_svm, X_test_svm, y_train_svm, y_test_svm = train_test_split(X, y_labels, test_size=0.2, random_state=42)
svm_model = SVC(kernel='linear', probability=True, random_state=42)
svm_model.fit(X_train_svm, y_train_svm)

# Print classes for SVM model
svm_classes = label_encoder.classes_
print("SVM Classes:", svm_classes)

y_pred_svm = svm_model.predict(X_test_svm)

print("SVM Accuracy:", accuracy_score(y_test_svm, y_pred_svm))
print("\nSVM Classification Report:\n", classification_report(
    y_test_svm, y_pred_svm, target_names=label_encoder.classes_, labels=np.arange(len(label_encoder.classes_))
))
joblib.dump(svm_model, 'hand_gesture_svm.joblib')  # Save to disk

# K-Nearest Neighbors Classifier
knn_model = KNeighborsClassifier(n_neighbors=5)
knn_model.fit(X_train_svm, y_train_svm)

# Print classes for KNN model
knn_classes = label_encoder.classes_
print("KNN Classes:", knn_classes)

y_pred_knn = knn_model.predict(X_test_svm)

print("KNN Accuracy:", accuracy_score(y_test_svm, y_pred_knn))
print("\nKNN Classification Report:\n", classification_report(
    y_test_svm, y_pred_knn, target_names=label_encoder.classes_, labels=np.arange(len(label_encoder.classes_))
))
joblib.dump(knn_model, 'hand_gesture_knn.joblib')  # Save to disk