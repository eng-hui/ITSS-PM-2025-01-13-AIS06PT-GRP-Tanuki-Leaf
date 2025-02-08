import cv2
import tkinter as tk
from PIL import Image, ImageTk
from .config import config
from .gesture import GestureLibrary
from .shapes import ShapeManager
from .mediapipe_utils import get_hands, get_drawing_utils
from .detection import detect_objects
from .gestureedit import GestureEditor  # Import GestureEditor

def launch_gesture_edit():
    # Launch GestureEditor
    editor = GestureEditor()
    editor.mainloop()

class Application:
    def __init__(self):
        self.config = config

        # Initialise Tkinter window.
        self.root = tk.Tk()
        self.root.title(self.config["ui"]["window_title"])
        self.root.geometry(self.config["ui"]["window_geometry"])
        self.root.grid_rowconfigure(0, weight=1)
        self.root.grid_columnconfigure(1, weight=1)

        # Left frame for controls.
        self.button_frame = tk.Frame(self.root)
        self.button_frame.grid(row=0, column=0, sticky="ns", padx=10, pady=10)

        tk.Button(self.button_frame, text="Triangle",
                  command=lambda: self.shape_manager.update_shape("triangle")).pack(pady=5, fill=tk.X)
        tk.Button(self.button_frame, text="Rectangle",
                  command=lambda: self.shape_manager.update_shape("rectangle")).pack(pady=5, fill=tk.X)
        tk.Button(self.button_frame, text="Circle",
                  command=lambda: self.shape_manager.update_shape("circle")).pack(pady=5, fill=tk.X)

        tk.Label(self.button_frame, text="Blur Intensity").pack(pady=5)
        self.blur_slider = tk.Scale(self.button_frame, from_=1, to=self.config["blur"]["max_value"],
                                    orient=tk.HORIZONTAL, command=self.update_blur)
        self.blur_slider.set(self.config["blur"]["default_slider"])
        self.blur_slider.pack(fill=tk.X)

        tk.Label(self.button_frame, text="Gesture Name").pack(pady=5)
        self.gesture_entry = tk.Entry(self.button_frame)
        self.gesture_entry.pack(pady=5, fill=tk.X)

        tk.Button(self.button_frame, text="Capture Gesture", command=self.capture_gesture_action) \
            .pack(pady=5, fill=tk.X)

        # Add the new button for launching GestureEdit
        tk.Button(self.button_frame, text="Launch GestureEdit", command=launch_gesture_edit) \
            .pack(pady=5, fill=tk.X)

        # Right frame for video display.
        self.video_frame = tk.Frame(self.root)
        self.video_frame.grid(row=0, column=1, sticky="nsew", padx=10, pady=10)
        self.lmain = tk.Label(self.video_frame)
        self.lmain.pack(fill=tk.BOTH, expand=True)

        # Initialise camera.
        self.cap = cv2.VideoCapture(self.config["camera"]["device"])
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.config["camera"]["frame_width"])
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.config["camera"]["frame_height"])

        # Initialise MediaPipe and drawing utilities.
        self.hands = get_hands()
        self.mp_drawing = get_drawing_utils()

        # Initialise gesture library.
        self.gesture_lib = GestureLibrary(self.config["gesture"]["file"],
                                          similarity_threshold=self.config["gesture"]["similarity_threshold"])

        # Initialise shape manager.
        self.shape_manager = ShapeManager(self.gesture_lib.library)

        # Set initial blur intensity.
        self.blur_intensity = max(1, int(self.blur_slider.get()) * 2 + 1)

    def update_blur(self, value):
        self.blur_intensity = max(1, int(value) * 2 + 1)

    def capture_gesture_action(self):
        gesture_name = self.gesture_entry.get().strip()
        if not gesture_name:
            print("No gesture name entered.")
            return

        ret, frame = self.cap.read()
        if not ret:
            print("Unable to read from camera.")
            return

        frame = cv2.flip(frame, 1)
        image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.hands.process(image_rgb)
        if not results.multi_hand_landmarks or not results.multi_handedness:
            print("No hand detected.")
            return

        hand_landmarks = results.multi_hand_landmarks[0]
        hand_label = results.multi_handedness[0].classification[0].label
        from .mediapipe_utils import extract_hand_vectors
        vectors = extract_hand_vectors(hand_landmarks)
        side = "Left" if hand_label.lower() == "left" else "Right"
        self.gesture_lib.add_gesture(side, gesture_name, vectors)
        print(f"Captured gesture '{gesture_name}' for {side} hand.")

    def show_frame(self):
        ret, frame = self.cap.read()
        if ret:
            frame = cv2.flip(frame, 1)
            frame = cv2.resize(frame, (self.config["camera"]["frame_width"], self.config["camera"]["frame_height"]))
            processed_frame = detect_objects(frame, self.shape_manager, self.gesture_lib,
                                             self.blur_intensity, self.hands, self.mp_drawing)
            frame_resized = cv2.resize(processed_frame, (self.lmain.winfo_width(), self.lmain.winfo_height()))
            cv2image = cv2.cvtColor(frame_resized, cv2.COLOR_BGR2RGB)
            img = Image.fromarray(cv2image)
            imgtk = ImageTk.PhotoImage(image=img)
            self.lmain.imgtk = imgtk
            self.lmain.configure(image=imgtk)
        self.lmain.after(30, self.show_frame)  # Increase the delay to reduce the update frequency

    def run(self):
        self.root.update_idletasks()
        self.show_frame()
        self.root.mainloop()

def start_ui():
    app = Application()
    app.run()
