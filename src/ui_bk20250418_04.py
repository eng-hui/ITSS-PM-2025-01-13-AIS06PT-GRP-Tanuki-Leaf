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
from streamdiffusion.image_utils import postprocess_image
import threading
import speech_recognition as sr
import time

def launch_gesture_edit():
    # Launch GestureEditor
    editor = GestureEditor()
    editor.mainloop()
    
class Application:
    def __init__(self):
        self.config = config

        self.diffusion_lock = threading.Lock()  # Lock for thread safety
        
        # Increase the window size.
        self.root = tk.Tk()
        self.root.title(self.config["ui"]["window_title"])
        self.root.geometry("1400x800+50+50")  # Expanded window size
        self.root.grid_rowconfigure(0, weight=1)
        self.root.grid_columnconfigure(1, weight=1)

        # Initialize speech recognition
        self.recognizer = sr.Recognizer()
        self.listening = False
        self.voice_active = False  # Flag to track if voice recognition is enabled
        self.activation_phrase = self.config.get("voice_recognition", {}).get(
        "activation_phrase", "activate")# Get activation phrase from config or use default
        self.listen_thread = None  # Keep track of the background listening thread
        
        # Load model
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
            
        # Add Voice toggle button with updated text
        self.voice_button = tk.Button(self.button_frame, text="Voice Recognition: OFF", 
                                    command=self.toggle_voice_recognition)
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
        self.stream = StreamDiffusion(
            self.pipe,
            # t_index_list=[16,18,20,22],
            t_index_list = self.config["diffusion"]["t_index_list"],
            torch_dtype=torch.float16,
            cfg_type = self.config["diffusion"]["cfg_type"] 
        )

        lora_path = self.config["diffusion"]["lora"]
        use_lora = self.config["diffusion"]["use_lora"]
        encoder_path = self.config["diffusion"]["encoder"]
        self.stream.load_lcm_lora()
        if use_lora:
            self.stream.pipe.load_lora_weights(lora_path)
        self.stream.fuse_lora()

        self.stream.enable_similar_image_filter()
        self.stream.vae = AutoencoderTiny.from_pretrained(encoder_path).to(
            device=self.pipe.device, dtype=self.pipe.dtype)
        self.pipe.enable_xformers_memory_efficient_attention()

        # Set initial blur intensity.
        self.blur_intensity = max(1, int(self.blur_slider.get()) * 2 + 1)

        # Flag to avoid overlapping diffusion tasks.
        self.diffusion_running = False
        default_key = list(self.prompt_array.keys())[0]
        self.diffusion_prompt = self.prompt_array[default_key] # set default prompt
        self.previous_prompt = None
        self.prepare_stream()
        
    def toggle_voice_recognition(self):
        """Toggle voice recognition on/off"""
        self.voice_active = not self.voice_active
        
        if self.voice_active:
            # Turn on continuous listening
            self.voice_button.config(text="Voice Recognition: ON", bg="green")
            self.voice_result_label.config(text="Waiting for 'Activate'...", fg="blue")
            
            # Start continuous listening in a separate thread
            if self.listen_thread is None or not self.listen_thread.is_alive():
                self.listen_thread = threading.Thread(target=self.continuous_listen, daemon=True)
                self.listen_thread.start()
        else:
            # Turn off continuous listening
            self.voice_button.config(text="Voice Recognition: OFF", bg="SystemButtonFace")
            self.voice_result_label.config(text="Voice recognition disabled", fg="gray")
            self.listening = False  # This will cause the continuous_listen loop to exit

    def continuous_listen(self):
        """Continuously listen for the activation phrase"""
        self.listening = True
        
        while self.listening and self.voice_active:
            try:
                with sr.Microphone() as source:
                    print("Listening for activation phrase...")
                    self.recognizer.adjust_for_ambient_noise(source, duration=0.5)
                    audio = self.recognizer.listen(source, phrase_time_limit=3)  # Short timeout for better responsiveness
                    
                try:
                    # Use Google's speech recognition service
                    text = self.recognizer.recognize_google(audio).lower()
                    print(f"Heard: {text}")
                    
                    # Check for activation phrase
                    if self.activation_phrase in text:
                        print("Activation phrase detected!")
                        # Update UI to show we're now listening for a command
                        self.root.after(0, lambda: self.voice_button.config(text="Listening for command...", bg="red"))
                        self.root.after(0, lambda: self.voice_result_label.config(
                            text="Activation phrase detected! Listening for command...", fg="green"))
                        
                        # Wait briefly before listening for the actual command
                        time.sleep(0.5)
                        self.listen_for_gesture_command()
                    
                except sr.UnknownValueError:
                    # No speech detected, continue listening
                    pass
                except sr.RequestError as e:
                    print(f"Could not request results; {e}")
                    self.root.after(0, lambda: self.voice_result_label.config(
                        text=f"API error: {str(e)[:30]}...", fg="red"))
                    # Wait before trying again
                    time.sleep(2)
                    
            except Exception as e:
                print(f"Error in continuous listening: {e}")
                # Short delay before retrying
                time.sleep(1)
                
        # If we exit the loop, make sure UI is reset if voice_active is still on
        if self.voice_active:
            self.root.after(0, lambda: self.voice_button.config(text="Voice Recognition: ON", bg="green"))
            self.root.after(0, lambda: self.voice_result_label.config(
                text="Waiting for 'Activate'...", fg="blue"))
                
    def listen_for_gesture_command(self):
        """Listen for a specific gesture command after activation phrase with enhanced recognition"""
        try:
            with sr.Microphone() as source:
                print("Listening for a gesture command...")
                self.recognizer.adjust_for_ambient_noise(source, duration=0.5)
                audio = self.recognizer.listen(source, timeout=5)
                
            try:
                # Use Google's speech recognition service
                text = self.recognizer.recognize_google(audio).lower()
                print(f"Originally recognized: {text}")
                
                # Apply post-processing to correct common misrecognitions
                text = self.improve_recognition(text)
                print(f"After processing: {text}")
                
                # Match recognized text with gestures
                found_match = False
                gesture_names = list(self.gesture_lib.get_all_gestures().keys())
                
                # First try direct matching
                for gesture_name in gesture_names:
                    # Simple check if the recognized text contains the gesture name
                    if gesture_name.lower() in text:
                        print(f"Matched gesture: {gesture_name}")
                        # Update result label with matched gesture
                        self.root.after(0, lambda g=gesture_name: self.voice_result_label.config(
                            text=f"Executing: {g}", fg="green"))
                        # Flag that this is a voice-activated change
                        self.voice_activated = True
                        # Update to this gesture in the main thread
                        self.root.after(0, lambda g=gesture_name: self.handle_gesture_change(g))
                        found_match = True
                        break
                
                # If no direct match, try fuzzy matching for short words
                if not found_match:
                    for gesture_name in gesture_names:
                        # For short words, check if the recognized text contains something similar
                        if len(gesture_name) <= 4 and self.is_similar_word(gesture_name.lower(), text):
                            print(f"Fuzzy matched gesture: {gesture_name}")
                            # Update result label with matched gesture
                            self.root.after(0, lambda g=gesture_name: self.voice_result_label.config(
                                text=f"Executing: {g} (fuzzy match)", fg="green"))
                            # Flag that this is a voice-activated change
                            self.voice_activated = True
                            # Update to this gesture in the main thread
                            self.root.after(0, lambda g=gesture_name: self.handle_gesture_change(g))
                            found_match = True
                            break
                
                if not found_match:
                    print(f"No matching gesture found for: {text}")
                    self.root.after(0, lambda: self.voice_result_label.config(
                        text=f"Command not recognized: {text}", fg="orange"))
                    
            except sr.UnknownValueError:
                print("Could not understand command")
                self.root.after(0, lambda: self.voice_result_label.config(
                    text="Could not understand command", fg="red"))
            
            except sr.RequestError as error:
                print(f"Could not request results; {error}")
                error_msg = str(error)
                self.root.after(0, lambda: self.voice_result_label.config(
                    text=f"Request error: {error_msg[:30]}", fg="red"))
                    
        except Exception as error:
            print(f"Error in command recognition: {error}")
            error_msg = str(error)
            self.root.after(0, lambda: self.voice_result_label.config(
                text=f"Error: {error_msg[:30]}...", fg="red"))
        
        finally:
            # Reset UI to continue listening for activation phrase
            if self.voice_active:
                self.root.after(0, lambda: self.voice_button.config(text="Voice Recognition: ON", bg="green"))
                self.root.after(0, lambda: self.voice_result_label.config(
                    text="Waiting for 'Activate'...", fg="blue"))

    def improve_recognition(self, text):
        """Apply custom corrections to improve speech recognition"""
        # Get corrections dictionary from config
        corrections = self.config.get("voice_recognition", {}).get("corrections", {})
        
        # Split text into words and check each one
        words = text.split()
        corrected_words = []
        
        for word in words:
            # Check if this word should be corrected
            if word in corrections:
                corrected_words.append(corrections[word])
            else:
                corrected_words.append(word)
        
        return " ".join(corrected_words)

    def is_similar_word(self, gesture_name, text):
        """Check if a short gesture name might be similar to something in the text"""
        words = text.split()
        
        # For very short words like "rem", check for partial matches
        for word in words:
            # If the word is very similar (differs by at most one character)
            if len(word) <= 5 and self.levenshtein_distance(gesture_name, word) <= 1:
                return True
                
            # Check for word endings (useful for plural forms, etc.)
            if len(word) >= len(gesture_name) and word.startswith(gesture_name):
                return True
        
        return False

    def levenshtein_distance(self, s1, s2):
        """Calculate the edit distance between two strings"""
        if len(s1) < len(s2):
            return self.levenshtein_distance(s2, s1)
        
        if len(s2) == 0:
            return len(s1)
        
        previous_row = range(len(s2) + 1)
        for i, c1 in enumerate(s1):
            current_row = [i + 1]
            for j, c2 in enumerate(s2):
                # Calculate insertions, deletions and substitutions
                insertions = previous_row[j + 1] + 1
                deletions = current_row[j] + 1
                substitutions = previous_row[j] + (c1 != c2)
                current_row.append(min(insertions, deletions, substitutions))
            previous_row = current_row
        
        return previous_row[-1]

    def handle_matched_gesture(self, gesture_name):
        """Common code for handling a matched gesture"""
        # Update result label with matched gesture
        self.root.after(0, lambda g=gesture_name: self.voice_result_label.config(
            text=f"Matched: {g}", fg="green"))
        # Flag that this is a voice-activated change
        self.voice_activated = True
        # Update to this gesture in the main thread
        self.root.after(0, lambda g=gesture_name: self.handle_gesture_change(g))
        
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

    def process_diffusion(self, frame):
        # Set flag so no new diffusion is started while running.
        self.diffusion_running = True
        try:
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
        finally:
            self.diffusion_running = False
            
            # Check if we need to update the prompt
            if getattr(self, 'prompt_needs_update', False):
                self.prompt_needs_update = False
                threading.Thread(target=self._prepare_stream_safe, daemon=True).start()
            
    def update_diffusion_label(self, imgtk):
        self.diffusion_label.config(image=imgtk)
        self.diffusion_label.image = imgtk  # Keep a reference.

    
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


        style_change_gesture = self.config["gesture"]["style_change_gesture"]
        if classified_gesture == style_change_gesture or self.isChangingPrompt:
            self.camera_label.config(borderwidth=5, relief="solid", highlightbackground="red", highlightcolor="red", highlightthickness=5)
            self.isChangingPrompt = True
        else:
            self.camera_label.config(borderwidth=0, relief="flat", highlightthickness=0)
            self.isChangingPrompt = False


        # For voice-activated changes, we need to handle them differently
        voice_activated = getattr(self, 'voice_activated', False)

        # If it's a voice command or we're in changing prompt mode and the prompt changed
        if (voice_activated and self.previous_prompt != self.diffusion_prompt) or \
        (self.isChangingPrompt and self.previous_prompt != self.diffusion_prompt and classified_gesture != 'None'):
            print("changing prompt:", self.diffusion_prompt, classified_gesture)
            
             # Create a "needs update" flag instead of requiring immediate action
            self.prompt_needs_update = True
        
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
        with self.diffusion_lock:
            self.diffusion_running = True
            try:
                self.prepare_stream()
            finally:
                self.diffusion_running = False
    
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
