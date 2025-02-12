import cv2
import numpy as np
import mediapipe as mp
from .explosion import create_explosion, update_particles
from .mediapipe_utils import extract_hand_vectors
from .utils import draw_text_with_outline

def detect_objects(frame, shape_manager, gesture_lib, blur_intensity, hands, mp_drawing):
    image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(image_rgb)
    h, w, _ = frame.shape

    if blur_intensity == 201:
        blurred_frame = np.zeros_like(frame)
    else:
        blurred_frame = cv2.GaussianBlur(frame, (blur_intensity, blur_intensity), 0)

    mask = np.zeros((h, w), dtype=np.uint8)
    x0, y0 = shape_manager.position
    explosion_triggered = False

    if results.multi_hand_landmarks and results.multi_handedness:
        for hlm, handedness in zip(results.multi_hand_landmarks, results.multi_handedness):
            collision = False
            for landmark in hlm.landmark:
                kx, ky = int(landmark.x * w), int(landmark.y * h)
                cv2.circle(mask, (kx, ky), 15, 255, -1)
                if x0 <= kx <= x0 + shape_manager.size and y0 <= ky <= y0 + shape_manager.size:
                    collision = True
                    break

            vectors = extract_hand_vectors(hlm)
            side = "Left" if handedness.classification[0].label.lower() == "left" else "Right"
            classified_gesture = gesture_lib.classify(vectors, side)
            if collision and classified_gesture == shape_manager.target_gesture:
                create_explosion(x0 + shape_manager.size // 2, y0 + shape_manager.size // 2)
                shape_manager.move(w, h)
                explosion_triggered = True
                break

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

    shape_manager.draw(final_frame)
    update_particles(final_frame)
    return final_frame
