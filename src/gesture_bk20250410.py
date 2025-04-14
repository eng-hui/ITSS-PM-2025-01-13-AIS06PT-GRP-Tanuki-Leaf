import json
import os
import numpy as np

class GestureLibrary:
    def __init__(self, file_path, similarity_threshold=0.85):
        self.file_path = file_path
        self.similarity_threshold = similarity_threshold
        self.library = {"Left": {}, "Right": {}}
        self.load()

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

    def classify(self, vectors, side):
        best_gesture = None
        best_score = 0.0
        for gesture_name, stored_samples in self.library.get(side, {}).items():
            for sample_vectors in stored_samples:
                if len(sample_vectors) == len(vectors):
                    sims = []
                    for v1, v2 in zip(sample_vectors, vectors):
                        sims.append(self.compute_cosine_similarity(v1, v2))
                    gesture_similarity = sum(sims) / len(sims)
                    if gesture_similarity > best_score:
                        best_score = gesture_similarity
                        best_gesture = gesture_name
        return best_gesture if best_score > self.similarity_threshold else "None"

    @staticmethod
    def compute_cosine_similarity(v1, v2):
        dot = np.dot(v1, v2)
        norm = (np.linalg.norm(v1) * np.linalg.norm(v2))
        return dot / norm if norm else 0
