import cv2
import tkinter as tk
from PIL import Image, ImageTk
from .config import config
from .gesture import GestureLibrary
from .shapes import ShapeManager
from .mediapipe_utils import get_hands, get_drawing_utils, extract_hand_vectors
from .detection import detect_objects
from .gestureedit import GestureEditor  # Import GestureEditor
import torch
from diffusers import AutoencoderTiny, StableDiffusionPipeline
from streamdiffusion import StreamDiffusion
from streamdiffusion.image_utils import postprocess_image
import threading
import numpy as np
def launch_gesture_edit():
    # Launch GestureEditor
    editor = GestureEditor()
    editor.mainloop()

class Application:
    def __init__(self):
        self.config = config

        # Increase the window size.
        self.root = tk.Tk()
        self.root.title(self.config["ui"]["window_title"])
        self.root.geometry("1400x800")  # Expanded window size
        self.root.grid_rowconfigure(0, weight=1)
        self.root.grid_columnconfigure(1, weight=1)

        # Diffusion prompt list and index. Pressing a trigger will step to the next prompt.
        self.prompt_array = [
            "gojo satoru, white hair, black eye mask",
            "A rugged bounty hunter with a cybernetic eye, wearing a tattered leather jacket and carrying a plasma rifle in a neon-lit dystopian city.",
            "A soft-spoken librarian with braided auburn hair, round glasses, and a vintage dress, carefully placing books on a towering wooden shelf.",
            "A street artist with vibrant green hair, wearing a paint-stained hoodie, spraying a giant mural of a phoenix on a city wall at night.",
            "A pirate captain with an eyepatch and a long, flowing coat, gripping the wheel of their ship as a storm brews on the horizon.",
            "A futuristic android with sleek silver plating, glowing blue eyes, and a human-like synthetic face, staring at their reflection in a mirror.",
            "A mysterious detective in a noir-style trench coat and fedora, smoking a cigar as they examine a cryptic note under a streetlamp.",
            "A medieval knight in intricately detailed armour, holding a massive sword and standing triumphantly in a battlefield at dawn.",
            "A punk rock musician with spiky red hair, ripped jeans, and multiple piercings, playing an electric guitar in front of a roaring crowd.",
            "A space explorer in a sleek astronaut suit, floating weightlessly inside a high-tech spaceship, staring at the distant glow of a nebula.",
            "A mischievous rogue with a dagger tucked into their belt, smirking as they flip a stolen coin in the dim light of a tavern.",
            "A cyberpunk hacker with a neon visor and a hooded jacket, typing furiously on a holographic keyboard as data streams across the screen."
        ]

        self.prompt_index = 0
        self.diffusion_prompt = self.prompt_array[self.prompt_index]
        self.negative_prompt = "low quality, bad quality, blurry, malformed"

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
        tk.Button(self.button_frame, text="Launch GestureEdit", command=launch_gesture_edit) \
            .pack(pady=5, fill=tk.X)

        # New diffusion label in the left panel with fixed size.
        self.diffusion_label = tk.Label(self.button_frame)
        self.diffusion_label.pack(pady=10)
        # Optionally, set a border or background to visually delimit the area:
        self.diffusion_label.config(borderwidth=2, relief="solid")

        # Right frame for live camera feed.
        self.video_frame = tk.Frame(self.root)
        self.video_frame.grid(row=0, column=1, sticky="nsew", padx=10, pady=10)
        self.video_frame.grid_rowconfigure(0, weight=1)
        self.video_frame.grid_columnconfigure(0, weight=1)

        # Label for live camera feed.
        self.camera_label = tk.Label(self.video_frame)
        self.camera_label.grid(row=0, column=0, sticky="nsew")

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

        # Initialise diffusion pipeline and StreamDiffusion.
        self.pipe = StableDiffusionPipeline.from_pretrained("KBlueLeaf/kohaku-v2.1").to(
            device=torch.device("cuda"),
            dtype=torch.float16,
        )
        self.stream = StreamDiffusion(
            self.pipe,
            t_index_list=[16,18,20,22],
            torch_dtype=torch.float16,
        )
        self.stream.load_lcm_lora()
        
        # load multiple lora, if trained properly, each lora should only has effect when the prompt contains keyword
        self.stream.pipe.load_lora_weights("data/gojo.safetensors")

        self.stream.fuse_lora()
        self.stream.vae = AutoencoderTiny.from_pretrained("madebyollin/taesd").to(
            device=self.pipe.device, dtype=self.pipe.dtype)
        self.pipe.enable_xformers_memory_efficient_attention()

        # Set initial blur intensity.
        self.blur_intensity = max(1, int(self.blur_slider.get()) * 2 + 1)

        # Flag to avoid overlapping diffusion tasks.
        self.diffusion_running = False
        self.prepare_stream()


    def prepare_stream(self):
        pil_image = None # fake image
        self.stream.prepare(self.diffusion_prompt, negative_prompt=self.negative_prompt, num_inference_steps=25)
        # Warm up steps.
        ret, frame = self.cap.read()
        if not ret:
            print("Unable to read from camera.")
            return
        image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        pil_image = Image.fromarray(image_rgb).resize((512, 512))
        self.stream(pil_image)
        for _ in range(4):
            _ = self.stream(pil_image)
        
        #self.text_to_image_pipeline.unet.set_attn_processor(AttnProcessor2_0())

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
        vectors = extract_hand_vectors(hand_landmarks)
        side = "Left" if hand_label.lower() == "left" else "Right"
        self.gesture_lib.add_gesture(side, gesture_name, vectors)
        print(f"Captured gesture '{gesture_name}' for {side} hand.")

        # Update to the next prompt in the array regardless of the gesture name.
        self.prompt_index = (self.prompt_index + 1) % len(self.prompt_array)
        self.diffusion_prompt = self.prompt_array[self.prompt_index]
        print(f"Updated diffusion prompt to: {self.diffusion_prompt}")

    def process_diffusion(self, frame):
        # Set flag so no new diffusion is started while running.
        self.diffusion_running = True
        # Convert frame and prepare PIL image.
        image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        pil_image = Image.fromarray(image_rgb).resize((512, 512))
        # Generate diffusion image.
        x_output = self.stream(pil_image)
        generated = postprocess_image(x_output, output_type="pil")[0]
        generated_resized = generated.resize((256, 256))
        imgtk = ImageTk.PhotoImage(generated_resized)
        # Schedule the UI update on the main thread.
        self.root.after(0, lambda: self.update_diffusion_label(imgtk))
        self.diffusion_running = False

    def update_diffusion_label(self, imgtk):
        self.diffusion_label.config(image=imgtk)
        self.diffusion_label.image = imgtk  # Keep a reference.

    def trigger_explosion(self):
        self.prompt_index = (self.prompt_index + 1) % len(self.prompt_array)
        self.diffusion_prompt = self.prompt_array[self.prompt_index]
        print(f"Explosion triggered, updated diffusion prompt to: {self.diffusion_prompt}")
        self.prepare_stream()

    def show_frame(self):
        ret, frame = self.cap.read()
        if ret:
            frame = cv2.flip(frame, 1)
            frame = cv2.resize(frame, (self.config["camera"]["frame_width"], self.config["camera"]["frame_height"]))
            if self.blur_intensity > 1:
                processed_frame = detect_objects(frame, self.shape_manager, self.gesture_lib,
                                                 self.blur_intensity, self.hands, self.mp_drawing,
                                                 trigger_explosion_callback=self.trigger_explosion)
            else:
                processed_frame = detect_objects(frame, self.shape_manager, self.gesture_lib,
                                                 None, self.hands, self.mp_drawing,
                                                 trigger_explosion_callback=self.trigger_explosion)
                                               
            # Start a diffusion process in the background if not already running.
            if not self.diffusion_running:
                threading.Thread(target=self.process_diffusion, args=(frame,), daemon=True).start()
            
            frame_resized = cv2.resize(processed_frame, (self.camera_label.winfo_width(), self.camera_label.winfo_height()))
            cv2image = cv2.cvtColor(frame_resized, cv2.COLOR_BGR2RGB)
            img = Image.fromarray(cv2image)
            imgtk = ImageTk.PhotoImage(image=img)
            self.camera_label.imgtk = imgtk
            self.camera_label.configure(image=imgtk)
        self.camera_label.after(30, self.show_frame)

    def run(self):
        self.root.update_idletasks()
        self.show_frame()
        self.root.mainloop()

def start_ui():
    app = Application()
    app.run()
