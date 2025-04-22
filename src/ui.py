import cv2
import tkinter as tk
from PIL import Image, ImageTk
from .config import config
from .gesture import GestureLibrary
from .mediapipe_utils import get_hands, get_drawing_utils, extract_hand_vectors
from .detection import detect_objects
from .gestureedit import GestureEditor  # Import GestureEditor
from .ml_gesture_recognition import MLGestureRecognition
import torch
from diffusers import AutoencoderTiny, StableDiffusionPipeline
from streamdiffusion import StreamDiffusion
#from .override import OverrideStreamDiffusion as StreamDiffusion
from streamdiffusion.image_utils import postprocess_image
import threading
import numpy as np
import speech_recognition as sr
from .controlnet import StableDiffusionControlNetPipeline, controlnet

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
        self.root.geometry("1400x800+50+50")  # Expanded window size
        self.root.grid_rowconfigure(0, weight=1)
        self.root.grid_columnconfigure(1, weight=1)
        self.use_reg_model = self.config["gesture"]["use_model"]
        self.reg_model = None
        self.reg_classes = []
        if self.use_reg_model:
            reg_classes = self.config["gesture"]["label_classes"]
            self.reg_model = MLGestureRecognition(self.config["gesture"]["model_path"],classes=reg_classes)


        # Load prompts and negative prompts from config
        self.prompt_array = self.config["prompts"]
        self.negative_prompt = self.config["negative_prompt"]
        self.diffusion_prompt = None
        self.isChangingPrompt = False

        # Left frame for controls.
        self.button_frame = tk.Frame(self.root)
        self.button_frame.grid(row=0, column=0, sticky="ns", padx=10, pady=10)

        # Initialize speech recognition
        self.recognizer = sr.Recognizer()
        self.listening = False

       # Load and display the image in the top-left corner
        image_path = self.config["ui"]["image_icon"] 
        try:
            img = Image.open(image_path)
            img_resized = img.resize((250, 250))  
            self.logo_image = ImageTk.PhotoImage(img_resized)

            self.image_label = tk.Label(
                self.button_frame,
                image=self.logo_image,
                bg="white",         # White background
                borderwidth=2,      # Border thickness
                relief="solid"      # Border style (solid line)
            )
            self.image_label.pack(pady=5, anchor='center')  # Centre it in the frame
        except Exception as e:
            print(f"Error loading image: {e}")


        tk.Label(self.button_frame, text="Blur Intensity",bg="black",fg="white",highlightbackground="black",
    highlightcolor="black").pack(pady=5, fill=tk.X)
        self.blur_slider = tk.Scale(self.button_frame, from_=1, to=self.config["blur"]["max_value"],
                                    orient=tk.HORIZONTAL, command=self.update_blur)
        self.blur_slider.set(self.config["blur"]["default_slider"])
        self.blur_slider.pack(fill=tk.X)

        tk.Label(self.button_frame, text="Gesture Name",bg="darkorange",fg="white",highlightbackground="black",
    highlightcolor="black").pack(pady=5, fill=tk.X)


        self.gesture_entry = tk.Entry(self.button_frame)
        self.gesture_entry.pack(pady=5, fill=tk.X)

        tk.Button(self.button_frame, text="Capture Gesture", command=self.capture_gesture_action) \
            .pack(pady=5, fill=tk.X)
        tk.Button(self.button_frame, text="Launch GestureEdit", command=launch_gesture_edit) \
            .pack(pady=5, fill=tk.X)

        # Add Voice button
        self.voice_button = tk.Button(self.button_frame, text="Voice Recognition", command=self.voice_recognition_action)
        self.voice_button.pack(pady=5, fill=tk.X)
                
        # Add voice result label for displaying
        self.voice_result_label = tk.Label(self.button_frame, text="", fg="blue", wraplength=200)
        self.voice_result_label.pack(pady=5, fill=tk.X)

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

        # Initialise diffusion pipeline and StreamDiffusion.
        base_model_path = self.config["diffusion"]["base_model"]
        self.pipe = StableDiffusionPipeline.from_pretrained(base_model_path).to(
            device=torch.device("cuda"),
            dtype=torch.float16,
        )
        # self.pipe = StableDiffusionControlNetPipeline.from_pretrained(
        #     r"KBlueLeaf/kohaku-v2.1", controlnet=controlnet, safety_checker=None, torch_dtype=torch.float16
        # ).to("cuda")
        # self.stream = StreamDiffusion(
        #     self.pipe,
        #     # t_index_list=[16,18,20,22],
        #     t_index_list = self.config["diffusion"]["t_index_list"],
        #     torch_dtype=torch.float16,
        #     cfg_type = self.config["diffusion"]["cfg_type"] 
        # )

        # lora_path = self.config["diffusion"]["lora"]
        # use_lora = self.config["diffusion"]["use_lora"]
        # encoder_path = self.config["diffusion"]["encoder"]
        # self.stream.load_lcm_lora()
        # if use_lora:
        #     self.stream.pipe.load_lora_weights(lora_path)
        # self.stream.pipe.unload_lora_weights()  
        # self.stream.fuse_lora()

        # self.stream.enable_similar_image_filter()
        # self.stream.vae = AutoencoderTiny.from_pretrained(encoder_path).to(
        #     device=self.pipe.device, dtype=self.pipe.dtype)
        # self.pipe.enable_xformers_memory_efficient_attention()
        

        # # Set initial blur intensity.
        # self.blur_intensity = max(1, int(self.blur_slider.get()) * 2 + 1)

        # # Flag to avoid overlapping diffusion tasks.
        # self.diffusion_running = False
        # default_key = list(self.prompt_array.keys())[0]
        # self.diffusion_prompt = self.prompt_array[default_key] # set default prompt
        # self.previous_prompt = None

        # Set initial blur intensity.
        self.classified_gesture = None
        self.blur_intensity = max(1, int(self.blur_slider.get()) * 2 + 1)
        self.first_flag = True

        # Flag to avoid overlapping diffusion tasks.
        self.diffusion_running = False
        default_key = list(self.prompt_array.keys())[0]
        self.diffusion_prompt = self.prompt_array[default_key] # set default prompt
        self.previous_prompt = None
        self.stream = StreamDiffusion(
            self.pipe,
            # t_index_list=[16,18,20,22],
            t_index_list = self.config["diffusion"]["t_index_list"],
            torch_dtype=torch.float16,
            cfg_type = self.config["diffusion"]["cfg_type"] 
        )

        self.prepare_stream()
        
    def voice_recognition_action(self):
        """Start voice recognition in a separate thread to avoid blocking UI"""
        if self.listening:
            # If already listening, do nothing
            return

        # Update button appearance to indicate listening
        self.voice_button.config(text="Listening...", bg="red")
        self.listening = True
        
        # Start listening in a thread
        threading.Thread(target=self.listen_for_gesture, daemon=True).start()
    
    def listen_for_gesture(self):
        """Listen for voice commands and match with gestures"""
        try:
            with sr.Microphone() as source:
                print("Listening for a gesture name...")
                # Adjust for ambient noise
                self.recognizer.adjust_for_ambient_noise(source, duration=0.5)
                # Listen for audio input
                audio = self.recognizer.listen(source, timeout=5)
                
            try:
                # Use Google's speech recognition service
                text = self.recognizer.recognize_google(audio).lower()
                print(f"Recognized: {text}")
                
                # Match recognized text with gestures
                found_match = False
                # Get the available gestures using the get_gestures method 
                # or access the correct attribute that contains gesture names
                gesture_names = list(self.gesture_lib.get_all_gestures().keys())
                
                for gesture_name in gesture_names:
                    # Simple check if the recognized text contains the gesture name
                    if gesture_name.lower() in text:
                        print(f"Matched gesture: {gesture_name}")
                        # Update result label with matched gesture
                        self.root.after(0, lambda g=gesture_name: self.voice_result_label.config(
                            text=f"Matched: {g}", fg="green"))
                        # Flag that this is a voice-activated change
                        self.voice_activated = True
                        # Update to this gesture in the main thread
                        self.root.after(0, lambda g=gesture_name: self.handle_gesture_change(g))
                        found_match = True
                        break
                
                if not found_match:
                    print(f"No matching gesture found for: {text}")
                    self.root.after(0, lambda: self.voice_result_label.config(
                        text="No matching gesture found", fg="orange"))
                    
            except sr.UnknownValueError:
                print("Could not understand audio")
                self.root.after(0, lambda: self.voice_result_label.config(
                    text="Could not understand audio", fg="red"))
            
            except sr.RequestError as error:
                print(f"Could not request results; {error}")
                # Store error message in a local variable before using in lambda
                error_msg = str(error)
                self.root.after(0, lambda: self.voice_result_label.config(
                    text=f"Request error: {error_msg[:30]}", fg="red"))
                    
        except Exception as error:
            print(f"Error in voice recognition: {error}")
            # Store error message in a local variable before using in lambda
            error_msg = str(error)
            self.root.after(0, lambda: self.voice_result_label.config(
                text=f"Error: {error_msg[:30]}...", fg="red"))
        
        finally:
            # Reset button appearance
            self.root.after(0, lambda: self.voice_button.config(text="Voice Recognition", bg="SystemButtonFace"))
            self.listening = False

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
        self.stream.pipe.unload_lora_weights()  
        lora_path = self.config["diffusion"]["lora"]
        use_lora = self.config["diffusion"]["use_lora"]
        encoder_path = self.config["diffusion"]["encoder"]
        print(f"load {self.classified_gesture}")
        self.stream.load_lcm_lora()
        if use_lora and (self.classified_gesture in ("satoru", "1", 1)):
            print("check lora loaded")
            self.stream.pipe.load_lora_weights(lora_path)
        else:
            pass
            #self.stream.pipe.load_lora_weights(lora_path)
        
        # self.stream.fuse_lora()

        if self.first_flag:
            self.stream.pipe.load_lora_weights(lora_path)
            self.stream.enable_similar_image_filter()
            self.stream.vae = AutoencoderTiny.from_pretrained(encoder_path).to(
                device=self.pipe.device, dtype=self.pipe.dtype)
            self.pipe.enable_xformers_memory_efficient_attention()
            self.first_flag = False

        pil_image = None # fake images
        print("xxxxxxxxxxxxxxxxxxxxxxxx")
        num_inference_steps = self.config["diffusion"]["num_inference_steps"]
        self.stream.prepare(self.diffusion_prompt, negative_prompt=self.negative_prompt, num_inference_steps=num_inference_steps)
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
        
        print("hhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhh")

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
        # Check if the frame is valid
        if not ret:
            print("Failed to read from camera.")
            return
        
        if ret:
            frame = cv2.flip(frame, 1)
            frame = cv2.resize(frame, (self.config["camera"]["frame_width"], self.config["camera"]["frame_height"]))
            
           
            # Continue with object detection or other processing.
            if self.blur_intensity > 1:
                processed_frame, classified_gesture = detect_objects(frame , self.gesture_lib,
                                                self.blur_intensity, self.hands, self.mp_drawing,use_reg_model=self.use_reg_model,reg_model=self.reg_model)
            else:
                processed_frame, classified_gesture = detect_objects(frame, self.gesture_lib,
                                                None, self.hands, self.mp_drawing,use_reg_model=self.use_reg_model,reg_model=self.reg_model)
                                                       
            self.handle_gesture_change(classified_gesture)
                    
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

    def handle_gesture_change(self, classified_gesture):
        if classified_gesture and classified_gesture != 'None':
            self.diffusion_prompt = self.prompt_array.get(classified_gesture, self.diffusion_prompt)
            self.classified_gesture = classified_gesture
            


        style_change_gesture = self.config["gesture"]["style_change_gesture"]
        if classified_gesture == style_change_gesture or self.isChangingPrompt:
            self.camera_label.config(borderwidth=5, relief="solid", highlightbackground="red", highlightcolor="red", highlightthickness=5)
            self.isChangingPrompt = True
        else:
            self.camera_label.config(borderwidth=0, relief="flat", highlightthickness=0)
            self.isChangingPrompt = False


        # For voice-activated changes, we need to handle them differently
        voice_activated = getattr(self, 'voice_activated', False)

        #If it's a voice command or we're in changing prompt mode and the prompt changed
        if (voice_activated and self.previous_prompt != self.diffusion_prompt) or \
        (self.isChangingPrompt and self.previous_prompt != self.diffusion_prompt and classified_gesture != 'None'):
            print(self.classified_gesture)
            print("changing prompt:", self.diffusion_prompt, classified_gesture)
            
            self.diffusion_running = False
            # Avoid conflicts with diffusion processing
            if not self.diffusion_running:
                # Run prepare_stream in a separate thread
                threading.Thread(target=self._prepare_stream_safe, daemon=True).start()
                
            self.previous_prompt = self.diffusion_prompt
            self.teleprompter_canvas.coords(self.teleprompter_text, self.teleprompter_canvas.winfo_width(), 15)
            
            # Reset flags
            if voice_activated:
                self.voice_activated = False
            else:
                self.isChangingPrompt = False

    def _prepare_stream_safe(self):
        """Thread-safe wrapper for prepare_stream"""
        # Set flag so diffusion knows we're preparing
        print("oreprare strean safe")
        self.diffusion_running = True
        try:
            self.prepare_stream()
        except Exception as e:
            print(e)
        finally:
            print("hello")
            self.diffusion_running = False
        
        

        if self.isChangingPrompt and self.previous_prompt != self.diffusion_prompt and classified_gesture != 'None':
            print("changing prompt x:", self.diffusion_prompt,classified_gesture)
            # change prompt here
            self.prepare_stream()
            self.previous_prompt = self.diffusion_prompt
            self.teleprompter_canvas.coords(self.teleprompter_text, self.teleprompter_canvas.winfo_width(), 15)
            self.isChangingPrompt = False


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
