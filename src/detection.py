import cv2
import numpy as np
import mediapipe as mp
from .explosion import create_explosion, update_particles
from .mediapipe_utils import extract_hand_vectors
from .utils import draw_text_with_outline
import time

def detect_objects(frame, gesture_lib, blur_intensity, hands, mp_drawing, trigger_diffusion_callback=None, trigger_explosion_callback=None, use_reg_model=False, reg_model=None):
    image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(image_rgb)
    h, w, _ = frame.shape

    if blur_intensity == 201:
        blurred_frame = np.zeros_like(frame)
    else:
        blurred_frame = cv2.GaussianBlur(frame, (blur_intensity, blur_intensity), 0)

    mask = np.zeros((h, w), dtype=np.uint8)
    explosion_triggered = False
    classified_gesture = None

    hand_info = []  # To store all relevant info per hand

    if results.multi_hand_landmarks and results.multi_handedness:
        for hlm, handedness in zip(results.multi_hand_landmarks, results.multi_handedness):
            hand_label = handedness.classification[0].label
            side = "Left" if hand_label.lower() == "left" else "Right"

            # Draw mask
            for landmark in hlm.landmark:
                kx, ky = int(landmark.x * w), int(landmark.y * h)
                cv2.circle(mask, (kx, ky), 15, 255, -1)

            vectors = extract_hand_vectors(hlm)

            if use_reg_model:
                start_time = time.time()
                gesture = reg_model.classify(vectors)
                end_time = time.time()
                print(f"Model Classification time: {end_time - start_time:.4f} seconds")
            else:
                start_time = time.time()
                gesture = gesture_lib.classify(vectors, side)
                end_time = time.time()
                print(f"Cosine Similarity time: {end_time - start_time:.4f} seconds")

            # Save everything for later use
            hand_info.append({
                'hlm': hlm,
                'side': side,
                'label': hand_label,
                'gesture': gesture
            })

            if side == "Left":
                classified_gesture = gesture  # Give priority to left hand
            elif classified_gesture is None:
                classified_gesture = gesture

    # Combine blurred and original frame based on mask
    mask_3channel = cv2.merge([mask, mask, mask])
    final_frame = np.where(mask_3channel == 255, frame, blurred_frame)

    # Now draw landmarks and gesture names
    for info in hand_info:
        hlm = info['hlm']
        hand_label = info['label']
        gesture = info['gesture']
        side = info['side']

        mp_drawing.draw_landmarks(final_frame, hlm, mp.solutions.hands.HAND_CONNECTIONS)

        x_label = int(hlm.landmark[0].x * w)
        y_label = int(hlm.landmark[0].y * h) - 10

        draw_text_with_outline(final_frame, f"{hand_label} Hand", (x_label, y_label),
                               cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,255), (0,0,0), 2)

        cv2.putText(final_frame, gesture, (x_label, y_label + 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), 2)

    return final_frame, classified_gesture
