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
        
        self.prompt_array = {
            '1': "A brilliant mad scientist in a high-tech laboratory, glowing neon lights, futuristic devices, intricate blueprints, steampunk goggles, determined expression, cinematic lighting, ultra-detailed, 8K",
            '2': "A hip hop rapper with gold chains, tattoos, sunglasses, baseball cap, baggy clothes, microphone, urban graffiti background, dynamic pose, cool attitude, 8K ultra-detailed",
            '3': "A wise and elegant librarian in an ancient, candle-lit library, surrounded by towering bookshelves, ancient scrolls, reading an enchanted book, warm golden lighting, mystical atmosphere, 8K ultra-detailed",
            'satoru':"Create an image of Gojo Satoru from Jujutsu Kaisen, featuring his signature white hair, blindfold or dark sunglasses, and his black high-collared outfit. Position him confidently, with a playful smile, against a dynamic background of blue energy and swirling cursed motifs, emphasizing his powerful and enigmatic presence, ultra-detailed, 8K",
            # 'satoru':"Create an image of Gojo Satoru, the powerful and enigmatic sorcerer from 'Jujutsu Kaisen.' Capture his iconic white hair, blindfold or sunglasses, and his confident, playful demeanor. He should be wearing his black high-collared outfit, standing against a backdrop that highlights his boundless energy and the mystic aura surrounding him, with hints of blue energy and swirling cursed energy motifs, ultra-detailed, 8K",
            '5': "Doraemon is a robotic cat from the future, with a round face, blue fur, no ears, and a pocket on his belly. He is often seen with his friend Nobita, a young Asian boy, ultra-detailed, 8K",
            'naruto':"Create an image of Naruto Uzumaki, the beloved ninja from 'Naruto.' Show him with his spiky blond hair, orange jumpsuit, and headband, striking a dynamic pose that reflects his energetic and determined personality. Place him in a vibrant, action-packed setting with swirling chakra energy and iconic ninja symbols, capturing the essence of his adventurous spirit and ninja skills, ultra-detailed, 8K",
            'luffy': "An image of Monkey D. Luffy from 'One Piece.' Depict him with his signature straw hat, red vest, and wide grin, standing confidently with his fists clenched. The background should feature the open sea or a battle scene, emphasizing his adventurous spirit and determination, ultra-detailed, 8K",
            'goku': "Picture of Goku from 'Dragon Ball Z.' Show him in his iconic orange gi, standing in a powerful pose with golden Super Saiyan hair and glowing energy surrounding him. The background should be an intense battle scene with energy blasts and dust clouds, ultra-detailed, 8K",
            'edward': "Drawing of Edward Elric from 'Fullmetal Alchemist.' Show him with his signature red coat, automail arm, and alchemy energy sparking from his hands. The background should have a mysterious transmutation circle glowing with blue light, ultra-detailed, 8K",    
            'rem': "Portrayal of Rem from 'Re:Zero.' Show her with her short blue hair, maid outfit, and a soft but determined expression. The background should be a fantasy setting with a hint of magical energy surrounding her, ultra-detailed, 8K",
    
        }
        
        # Default values
        self.diffusion_prompt = None

        self.negative_prompt = (
            "low quality, bad quality, blurry, distorted, malformed hands, unnatural anatomy, overexposed, "
            "mutated faces, extra limbs, low resolution, poorly drawn, ugly, deformed, watermark, artifacts, "
            "low contrast, washed out colors, pixelated, grainy, noisy, out of focus, poorly lit, "
            "overexposed highlights, underexposed shadows, incorrect proportions, awkward poses, "
            "unrealistic textures, flat lighting, lack of detail, boring composition, cluttered background"
        )

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

        # Teleprompter banner
        self.teleprompter_canvas = tk.Canvas(self.video_frame, height=30, bg="black")
        self.teleprompter_canvas.grid(row=1, column=0, sticky="ew")
        self.teleprompter_text = self.teleprompter_canvas.create_text(
            self.teleprompter_canvas.winfo_width(), 15, text="", fill="white", anchor="w"
        )
        
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
        self.stream.pipe.load_lora_weights("lora/satorugojo-10.safetensors")
        self.stream.fuse_lora()
        self.stream.enable_similar_image_filter()
        self.stream.vae = AutoencoderTiny.from_pretrained("madebyollin/taesd").to(
            device=self.pipe.device, dtype=self.pipe.dtype)
        self.pipe.enable_xformers_memory_efficient_attention()

        # Set initial blur intensity.
        self.blur_intensity = max(1, int(self.blur_slider.get()) * 2 + 1)

        # Flag to avoid overlapping diffusion tasks.
        self.diffusion_running = False
        self.diffusion_prompt = self.prompt_array['1'] # set default prompt
        self.previous_prompt = None
        self.prepare_stream()

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
            for _ in range(2):
                _ = self.stream(pil_image)

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
        # self.prompt_index = (self.prompt_index + 1) % len(self.prompt_array)
        # self.diffusion_prompt = self.prompt_array[self.prompt_index]
        # print(f"Explosion triggered, updated diffusion prompt to: {self.diffusion_prompt}")
        print("placeholder")
    
    def show_frame(self):
        ret, frame = self.cap.read()
        # Check if the frame is valid
        if not ret:
            print("Failed to read from camera.")
            return
        
        if ret:
            frame = cv2.flip(frame, 1)
            frame = cv2.resize(frame, (self.config["camera"]["frame_width"], self.config["camera"]["frame_height"]))
            
           
            # Continue with object detection or other processing.
            if self.blur_intensity > 1:
                processed_frame, classified_gesture = detect_objects(frame, self.shape_manager, self.gesture_lib,
                                                self.blur_intensity, self.hands, self.mp_drawing,
                                                trigger_explosion_callback=self.trigger_explosion)
            else:
                processed_frame, classified_gesture = detect_objects(frame, self.shape_manager, self.gesture_lib,
                                                None, self.hands, self.mp_drawing,
                                                trigger_explosion_callback=self.trigger_explosion)
                                                       



            
            if classified_gesture and classified_gesture != 'None':
                self.diffusion_prompt = self.prompt_array.get(classified_gesture, self.diffusion_prompt)
            else:
                classified_gesture = '1'
                self.diffusion_prompt = self.prompt_array.get(classified_gesture, self.diffusion_prompt)

            if self.previous_prompt != self.diffusion_prompt:
                self.prepare_stream()
            
            self.previous_prompt = self.diffusion_prompt
                    
            #  by array 
            # self.diffusion_prompt = self.prompt_array[self.prompt_index]
            
            # print("d prompt:", self.diffusion_prompt)
        
            # Start a diffusion process in the background if not already running.
            if not self.diffusion_running:
                # print("Start diffusion check")
                threading.Thread(target=self.process_diffusion, args=(frame,), daemon=True).start()
            
            
            # Resize frame to fit camera label.
            frame_resized = cv2.resize(processed_frame, (self.camera_label.winfo_width(), self.camera_label.winfo_height()))
            cv2image = cv2.cvtColor(frame_resized, cv2.COLOR_BGR2RGB)
            img = Image.fromarray(cv2image)
            imgtk = ImageTk.PhotoImage(image=img)
            self.camera_label.imgtk = imgtk
            self.camera_label.configure(image=imgtk)
            
            # Update teleprompter text
            self.update_teleprompter_text(self.diffusion_prompt)
            
            
        self.camera_label.after(30, self.show_frame)

    def update_teleprompter_text(self, text):
        self.teleprompter_canvas.itemconfig(self.teleprompter_text, text=text)
        self.teleprompter_canvas.coords(self.teleprompter_text, self.teleprompter_canvas.winfo_width(), 15)
        self.scroll_teleprompter()

    def scroll_teleprompter(self):
        x, y = self.teleprompter_canvas.coords(self.teleprompter_text)
        if x + self.teleprompter_canvas.bbox(self.teleprompter_text)[2] > 0:
            # self.teleprompter_canvas.move(self.teleprompter_text, -2, 0)
            # self.teleprompter_canvas.after(50, self.scroll_teleprompter)
            self.teleprompter_canvas.move(self.teleprompter_text, -5, 0)  # Increase the move distance
            self.teleprompter_canvas.after(30, self.scroll_teleprompter)  # Decrease the delay

    def run(self):
        self.root.update_idletasks()
        self.show_frame()        
        self.root.mainloop()

def start_ui():
    app = Application()
    app.run()
