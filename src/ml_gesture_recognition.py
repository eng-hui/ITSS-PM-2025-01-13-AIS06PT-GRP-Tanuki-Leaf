import joblib
import numpy as np

class MLGestureRecognition:
    def __init__(self, model_path, target_length=40,classes=None):
        self.model = joblib.load(model_path)
        self.target_length = target_length
        self.classes = classes

    def preprocess_vector(self, vector):
        # Convert the list to a NumPy array
        vector = np.array(vector)
        # Remove zero values
        filtered_sample = vector[vector != 0]
        
        # Pad or truncate to the target length
        if filtered_sample.size < self.target_length:
            pad_width = self.target_length - filtered_sample.size
            modified_sample = np.pad(filtered_sample, (0, pad_width), mode='constant')
        elif filtered_sample.size > self.target_length:
            modified_sample = filtered_sample[:self.target_length]
        else:
            modified_sample = filtered_sample
        
        return modified_sample

    def classify(self, vector):
        if vector is None or len(vector) == 0:
            return "None"
        
        input_data = self.preprocess_vector(vector).reshape(1, -1)  # Reshape for model input
        prediction = self.model.predict(input_data)
        
        return self.classes[np.argmax(prediction[0])]