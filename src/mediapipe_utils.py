import mediapipe as mp
import numpy as np
import math

def get_hands():
    mp_hands = mp.solutions.hands
    return mp_hands.Hands(
        static_image_mode=False,
        max_num_hands=2,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5
    )

def get_drawing_utils():
    return mp.solutions.drawing_utils

def normalise_landmarks(landmarks):
    """
    Normalise landmarks:
    1. Convert to a numpy array.
    2. Translate so that the wrist (index 0) is at (0,0).
    3. Rotate so the line from wrist to middle knuckle (index 9) aligns with the x-axis.
    4. Scale so that the bounding box diagonal equals 1.
    """
    points = np.array([[lm.x, lm.y] for lm in landmarks], dtype=np.float32)
    wrist = points[0].copy()
    points -= wrist
    ref = points[9]
    angle = math.atan2(ref[1], ref[0])
    cosA, sinA = math.cos(-angle), math.sin(-angle)
    R = np.array([[cosA, -sinA],
                  [sinA,  cosA]], dtype=np.float32)
    points = points @ R.T
    min_xy = points.min(axis=0)
    max_xy = points.max(axis=0)
    diag = np.linalg.norm(max_xy - min_xy)
    if diag > 1e-9:
        points /= diag
    return points

def extract_hand_vectors(hand_landmarks):
    """
    Extract vectors between all pairs of normalized landmarks.
    """
    norm_points = normalise_landmarks(hand_landmarks.landmark)
    num_points = len(norm_points)
    vectors = []
    for i in range(num_points):
        for j in range(i + 1, num_points):
            v = norm_points[j] - norm_points[i]
            vectors.append(v)
    return vectors
