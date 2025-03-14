import json
import os
import numpy as np
import torch  
import joblib
import yaml 

class GestureLibrary:
    def __init__(self, file_path, similarity_threshold=0.85):
        self.file_path = file_path
        self.similarity_threshold = similarity_threshold
        self.library = {"Left": {}, "Right": {}}
        self.config = self.load_config("config.yaml")
        # Model cache to avoid reloading models
        self.model_cache = {}
        # New: target length for feature vector (adjust as needed)
        self.target_length = 40  
        self.load()

    def load_config(self, config_path):
        """Load configuration from a YAML file"""
        try:
            with open(config_path, 'r') as f:
                config = yaml.safe_load(f)
            return config
        except Exception as e:
            print(f"Error loading config from {config_path}: {e}")
            return {}
        
    def load(self):
        if os.path.exists(self.file_path):
            with open(self.file_path, "r") as f:
                data = json.load(f)
            self.library = {"Left": {}, "Right": {}}
            for side in data:
                for gesture, samples in data[side].items():
                    self.library[side][gesture] = []
                    for sample in samples:
                        self.library[side][gesture].append([np.array(vec) for vec in sample])
        else:
            self.library = {"Left": {}, "Right": {}}

    def save(self):
        lib_to_save = {"Left": {}, "Right": {}}
        for side in self.library:
            for gesture, samples in self.library[side].items():
                lib_to_save[side][gesture] = []
                for sample in samples:
                    lib_to_save[side][gesture].append([vec.tolist() for vec in sample])
        with open(self.file_path, "w") as f:
            json.dump(lib_to_save, f)

    def add_gesture(self, side, gesture_name, vectors):
        if gesture_name not in self.library[side]:
            self.library[side][gesture_name] = []
        self.library[side][gesture_name].append(vectors)
        self.save()

    # def get_model(self, model_name):
            
    #     try:
    #         # For PyTorch model (NN)
    #         if model_name == "NN" and model_path.endswith(".pth"):
    #             # For PyTorch NN, we need to define and load the model architecture
    #             # This would require your NN model class from gesture_model_pytorch.py
    #             # For now, we'll return None until model architecture is defined
    #             print(f"PyTorch model loading not fully implemented for: {model_path}")
    #             return None
    #         else:
    #             # For scikit-learn models (.joblib files)
    #             loaded_model = joblib.load(model_path)
    #             self.model_cache[model_name] = loaded_model
    #             print(f"Loaded model {model_name} from {model_path}")
    #             return loaded_model
    #     except Exception as e:
    #         print(f"Error loading model {model_name} from {model_path}: {e}")
    #         return None    

    def classify(self, vectors,side):
        # If a recognition model is provided, use it.
        # """Load and cache a model based on the model name selected in UI"""
        # if model_name in self.model_cache:
        #     return self.model_cache[model_name]
        
        # if not self.config or not self.config.get("model"):
        #     return None
            
        # model_path = None
        # if model_name == "MLP":
        #     model_path = self.config["model"].get("model_mlp")
        # elif model_name == "Random Forest":
        #     model_path = self.config["model"].get("model_rf")
        # elif model_name == "Decision Tree":
        #     model_path = self.config["model"].get("model_dt")
        # elif model_name == "SVM":
        #     model_path = self.config["model"].get("model_svm")
        # elif model_name == "KNN":
        #     model_path = self.config["model"].get("model_knn") 
        # elif model_name == "NN":
        #     model_path = self.config["model"].get("model_nn")
            
        # if not model_path:
        #     return None
        
        self.recognition_model = joblib.load("hand_gesture_mlp.joblib")
        if self.recognition_model is not None: 
            # Flatten the list of vectors into a 1D feature vector.
            flat_feature = np.array(vectors).flatten()
            # Pad or truncate to target_length.
            if flat_feature.shape[0] < self.target_length:
                pad_width = self.target_length - flat_feature.shape[0]
                flat_feature = np.pad(flat_feature, (0, pad_width), mode='constant')
            elif flat_feature.shape[0] > self.target_length:
                flat_feature = flat_feature[:self.target_length]
            # Predict the gesture label using the model. Assumes model accepts a 2D array.
            prediction = self.recognition_model.predict([flat_feature])[0]
            return prediction
        # else:
        #     # Fallback to cosine similarity-based matching.
        #     best_gesture = None
        #     best_score = 0.0
        #     for gesture_name, stored_samples in self.library.get(side, {}).items():
        #         for sample_vectors in stored_samples:
        #             if len(sample_vectors) == len(vectors):
        #                 sims = []
        #                 for v1, v2 in zip(sample_vectors, vectors):
        #                     sims.append(self.compute_cosine_similarity(v1, v2))
        #                 gesture_similarity = sum(sims) / len(sims)
        #                 if gesture_similarity > best_score:
        #                     best_score = gesture_similarity
        #                     best_gesture = gesture_name
        #     return best_gesture if best_score > self.similarity_threshold else "None"

    @staticmethod
    def compute_cosine_similarity(v1, v2):
        dot = np.dot(v1, v2)
        norm = (np.linalg.norm(v1) * np.linalg.norm(v2))
        return dot / norm if norm else 0
