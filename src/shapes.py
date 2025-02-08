import cv2
import numpy as np
import random
from .config import config
from .utils import draw_text_with_outline

class ShapeManager:
    def __init__(self, gesture_library):
        self.shape = config["shape"]["default"]
        self.position = tuple(config["shape"]["default_position"])
        self.size = config["shape"]["default_size"]
        self.target_gesture = "None"
        self.gesture_library = gesture_library  # Expected to be a dict

    def move(self, screen_width, screen_height):
        margin = 50
        new_x = random.randint(margin, screen_width - self.size - margin)
        new_y = random.randint(margin, screen_height - self.size - margin)
        self.position = (new_x, new_y)
        # Update target gesture randomly from available gestures.
        all_gestures = list(self.gesture_library.get("Left", {}).keys()) + \
                       list(self.gesture_library.get("Right", {}).keys())
        self.target_gesture = random.choice(all_gestures) if all_gestures else "None"

    def update_shape(self, new_shape):
        self.shape = new_shape

    def draw(self, frame):
        x, y = self.position
        colour = (random.randint(50, 255), random.randint(50, 255), random.randint(50, 255))
        thickness = 2
        if self.shape == "rectangle":
            cv2.rectangle(frame, (x, y), (x + self.size, y + self.size), colour, thickness)
        elif self.shape == "triangle":
            pts = np.array([[x + self.size // 2, y],
                            [x, y + self.size],
                            [x + self.size, y + self.size]])
            cv2.polylines(frame, [pts], isClosed=True, color=colour, thickness=thickness)
        elif self.shape == "circle":
            cv2.circle(frame, (x + self.size // 2, y + self.size // 2), self.size // 2, colour, thickness)

        # Draw target gesture text centred in the shape.
        text = self.target_gesture
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 1
        text_thickness = 2
        text_size, _ = cv2.getTextSize(text, font, font_scale, text_thickness)
        text_x = x + (self.size - text_size[0]) // 2
        text_y = y + (self.size + text_size[1]) // 2
        draw_text_with_outline(frame, text, (text_x, text_y), font, font_scale, (255, 255, 255), (0, 0, 0), text_thickness)
        return frame
