import cv2
import numpy as np
import mediapipe as mp
from .explosion import create_explosion, update_particles
from .mediapipe_utils import extract_hand_vectors
from .utils import draw_text_with_outline

def detect_objects(frame, gesture_lib, blur_intensity, hands, mp_drawing, trigger_diffusion_callback=None, trigger_explosion_callback=None):
    image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(image_rgb)
    h, w, _ = frame.shape

    if blur_intensity == 201:
        blurred_frame = np.zeros_like(frame)
    else:
        blurred_frame = cv2.GaussianBlur(frame, (blur_intensity, blur_intensity), 0)

    mask = np.zeros((h, w), dtype=np.uint8)
    explosion_triggered = False
    classified_gesture = None  # Initialize classified_gesture

    left_hand_gesture = None
    right_hand_gesture = None

    if results.multi_hand_landmarks and results.multi_handedness:
        for hlm, handedness in zip(results.multi_hand_landmarks, results.multi_handedness):
            for landmark in hlm.landmark:
                kx, ky = int(landmark.x * w), int(landmark.y * h)
                cv2.circle(mask, (kx, ky), 15, 255, -1)

            vectors = extract_hand_vectors(hlm)
            side = "Left" if handedness.classification[0].label.lower() == "left" else "Right"
            gesture = gesture_lib.classify(vectors, side)
            
            if side == "Left":
                left_hand_gesture = gesture
            else:
                right_hand_gesture = gesture

            # If the special trigger gesture is detected, invoke the diffusion callback
            if gesture == "66666" and trigger_diffusion_callback is not None:
                trigger_diffusion_callback(frame)  # send raw frame for diffusion

    # Check for the specific gesture combination for naruto
    if left_hand_gesture == "naruto" and right_hand_gesture == "naruto":
        classified_gesture = "naruto"
    else:
        # For other gestures, use either left or right hand gesture
        if left_hand_gesture and left_hand_gesture != "naruto" :
            classified_gesture = left_hand_gesture
        elif right_hand_gesture and right_hand_gesture != "naruto":
            classified_gesture = right_hand_gesture
            
    mask_3channel = cv2.merge([mask, mask, mask])
    final_frame = np.where(mask_3channel == 255, frame, blurred_frame)

    if results.multi_hand_landmarks and results.multi_handedness:
        for hlm, handedness in zip(results.multi_hand_landmarks, results.multi_handedness):
            mp_drawing.draw_landmarks(final_frame, hlm, mp.solutions.hands.HAND_CONNECTIONS)
            hand_label = handedness.classification[0].label
            x_label = int(hlm.landmark[0].x * w)
            y_label = int(hlm.landmark[0].y * h) - 10
            draw_text_with_outline(final_frame, f"{hand_label} Hand", (x_label, y_label),
                                   cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,255), (0,0,0), 2)
            vectors = extract_hand_vectors(hlm)
            side = "Left" if hand_label.lower() == "left" else "Right"
            gesture_name = gesture_lib.classify(vectors, side)
            cv2.putText(final_frame, gesture_name,
                        (x_label, y_label + 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), 2)

    update_particles(final_frame)
    return final_frame, classified_gesture