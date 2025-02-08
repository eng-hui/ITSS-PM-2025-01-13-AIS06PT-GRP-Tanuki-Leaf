import tkinter as tk
from tkinter import messagebox, simpledialog
import json
import os
import numpy as np

GESTURE_FILE = "gesture_library.json"

def load_gestures():
    if os.path.exists(GESTURE_FILE):
        with open(GESTURE_FILE, "r") as f:
            data = json.load(f)
        return data
    else:
        return {"Left": {}, "Right": {}}

def save_gestures(data):
    with open(GESTURE_FILE, "w") as f:
        json.dump(data, f, indent=4)

class GestureEditor(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("Gesture Editor with 2D View")
        self.geometry("800x600")
        self.data = load_gestures()
        self.selected_side = None  # "Left" or "Right"
        self.selected_gesture = None
        self.selected_sample = None  # a sample is a list of 5 vectors (each a 2-element list)
        self.create_widgets()
        self.refresh_lists()

    def create_widgets(self):
        top_frame = tk.Frame(self)
        top_frame.pack(side=tk.TOP, fill=tk.BOTH, expand=True, padx=10, pady=10)

        # Left gestures section
        left_frame = tk.Frame(top_frame)
        left_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=10, pady=10)
        tk.Label(left_frame, text="Left Gestures").pack()
        self.left_listbox = tk.Listbox(left_frame)
        self.left_listbox.pack(fill=tk.BOTH, expand=True)
        self.left_listbox.bind("<<ListboxSelect>>", self.on_left_select)
        left_btn_frame = tk.Frame(left_frame)
        left_btn_frame.pack(pady=5)
        tk.Button(left_btn_frame, text="Rename", command=self.rename_left).pack(side=tk.LEFT, padx=5)
        tk.Button(left_btn_frame, text="Delete", command=self.delete_left).pack(side=tk.LEFT, padx=5)

        # Right gestures section
        right_frame = tk.Frame(top_frame)
        right_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=10, pady=10)
        tk.Label(right_frame, text="Right Gestures").pack()
        self.right_listbox = tk.Listbox(right_frame)
        self.right_listbox.pack(fill=tk.BOTH, expand=True)
        self.right_listbox.bind("<<ListboxSelect>>", self.on_right_select)
        right_btn_frame = tk.Frame(right_frame)
        right_btn_frame.pack(pady=5)
        tk.Button(right_btn_frame, text="Rename", command=self.rename_right).pack(side=tk.LEFT, padx=5)
        tk.Button(right_btn_frame, text="Delete", command=self.delete_right).pack(side=tk.LEFT, padx=5)

        # Detail frame for gesture info and 2D view
        detail_frame = tk.Frame(self)
        detail_frame.pack(side=tk.BOTTOM, fill=tk.BOTH, expand=False, padx=10, pady=10)
        self.detail_label = tk.Label(detail_frame, text="Select a gesture to view details", anchor="w")
        self.detail_label.pack(side=tk.TOP, fill=tk.X, pady=5)
        self.canvas = tk.Canvas(detail_frame, width=300, height=300, bg="white")
        self.canvas.pack(side=tk.TOP, pady=5)

        # Refresh button
        tk.Button(self, text="Refresh", command=self.refresh_lists).pack(pady=5)

    def refresh_lists(self):
        self.data = load_gestures()
        self.left_listbox.delete(0, tk.END)
        self.right_listbox.delete(0, tk.END)
        for gesture, samples in self.data.get("Left", {}).items():
            display_text = f"{gesture} ({len(samples)} sample{'s' if len(samples) != 1 else ''})"
            self.left_listbox.insert(tk.END, display_text)
        for gesture, samples in self.data.get("Right", {}).items():
            display_text = f"{gesture} ({len(samples)} sample{'s' if len(samples) != 1 else ''})"
            self.right_listbox.insert(tk.END, display_text)
        self.detail_label.config(text="Select a gesture to view details")
        self.clear_canvas()
        self.selected_side = None
        self.selected_gesture = None
        self.selected_sample = None

    def clear_canvas(self):
        self.canvas.delete("all")

    def draw_sample_on_canvas(self, sample):
        """
        Render the gesture sample correctly.
        Each sample is a list of 5 vectors.
        We draw each vector as an arrow starting from the canvas center.
        The scale is chosen so that the longest arrow fits within the canvas with a margin.
        To correct orientation (upside down and left/right flipped), we transform each vector.
        """
        self.clear_canvas()
        if not sample or len(sample) < 5:
            return

        # Canvas parameters
        canvas_width = int(self.canvas["width"])
        canvas_height = int(self.canvas["height"])
        center = np.array([canvas_width / 2, canvas_height / 2])
        margin = 20

        # Convert sample (list of 5 vectors) to a NumPy array of shape (5, 2)
        vectors = np.array(sample, dtype=np.float32)
        # Determine scaling factor so the longest arrow fits within the canvas.
        magnitudes = np.linalg.norm(vectors, axis=1)
        max_mag = np.max(magnitudes)
        if max_mag == 0:
            scale = 1.0
        else:
            scale = (min(canvas_width, canvas_height) / 2 - margin) / max_mag

        colors = ["red", "green", "blue", "orange", "purple"]
        finger_labels = ["Thumb", "Index", "Middle", "Ring", "Pinky"]

        for i, vec in enumerate(vectors):
            # Transform vector to correct orientation: invert both x and y.
            transformed_vec = np.array([-vec[0], -vec[1]])
            end_point = center + scale * transformed_vec
            # Draw an arrow from the center to the end point.
            self.canvas.create_line(center[0], center[1], end_point[0], end_point[1],
                                    arrow=tk.LAST, fill=colors[i], width=3)
            # Place a label near the arrow tip.
            self.canvas.create_text(end_point[0], end_point[1],
                                    text=finger_labels[i],
                                    fill=colors[i],
                                    font=("Arial", 10, "bold"))

    def on_left_select(self, event):
        selection = self.left_listbox.curselection()
        if selection:
            index = selection[0]
            display_text = self.left_listbox.get(index)
            gesture_name = display_text.split(" (")[0]
            samples = self.data.get("Left", {}).get(gesture_name, [])
            self.selected_side = "Left"
            self.selected_gesture = gesture_name
            self.selected_sample = samples[0] if samples else None
            self.update_detail_label()

    def on_right_select(self, event):
        selection = self.right_listbox.curselection()
        if selection:
            index = selection[0]
            display_text = self.right_listbox.get(index)
            gesture_name = display_text.split(" (")[0]
            samples = self.data.get("Right", {}).get(gesture_name, [])
            self.selected_side = "Right"
            self.selected_gesture = gesture_name
            self.selected_sample = samples[0] if samples else None
            self.update_detail_label()

    def update_detail_label(self):
        if self.selected_side and self.selected_gesture:
            samples = self.data.get(self.selected_side, {}).get(self.selected_gesture, [])
            info = f"Side: {self.selected_side}\nGesture: {self.selected_gesture}\nSamples: {len(samples)}"
            self.detail_label.config(text=info)
            if self.selected_sample:
                self.draw_sample_on_canvas(self.selected_sample)
            else:
                self.clear_canvas()

    def rename_left(self):
        selection = self.left_listbox.curselection()
        if not selection:
            messagebox.showinfo("Info", "Please select a gesture to rename.")
            return
        index = selection[0]
        display_text = self.left_listbox.get(index)
        old_name = display_text.split(" (")[0]
        new_name = simpledialog.askstring("Rename", "Enter new name for the gesture:", initialvalue=old_name)
        if new_name and new_name != old_name:
            if new_name in self.data.get("Left", {}):
                messagebox.showerror("Error", "A gesture with this name already exists.")
                return
            self.data["Left"][new_name] = self.data["Left"].pop(old_name)
            save_gestures(self.data)
            self.refresh_lists()

    def rename_right(self):
        selection = self.right_listbox.curselection()
        if not selection:
            messagebox.showinfo("Info", "Please select a gesture to rename.")
            return
        index = selection[0]
        display_text = self.right_listbox.get(index)
        old_name = display_text.split(" (")[0]
        new_name = simpledialog.askstring("Rename", "Enter new name for the gesture:", initialvalue=old_name)
        if new_name and new_name != old_name:
            if new_name in self.data.get("Right", {}):
                messagebox.showerror("Error", "A gesture with this name already exists.")
                return
            self.data["Right"][new_name] = self.data["Right"].pop(old_name)
            save_gestures(self.data)
            self.refresh_lists()

    def delete_left(self):
        selection = self.left_listbox.curselection()
        if not selection:
            messagebox.showinfo("Info", "Please select a gesture to delete.")
            return
        index = selection[0]
        display_text = self.left_listbox.get(index)
        gesture_name = display_text.split(" (")[0]
        confirm = messagebox.askyesno("Confirm Delete",
                                      f"Are you sure you want to delete gesture '{gesture_name}' from Left gestures?")
        if confirm:
            del self.data["Left"][gesture_name]
            save_gestures(self.data)
            self.refresh_lists()

    def delete_right(self):
        selection = self.right_listbox.curselection()
        if not selection:
            messagebox.showinfo("Info", "Please select a gesture to delete.")
            return
        index = selection[0]
        display_text = self.right_listbox.get(index)
        gesture_name = display_text.split(" (")[0]
        confirm = messagebox.askyesno("Confirm Delete",
                                      f"Are you sure you want to delete gesture '{gesture_name}' from Right gestures?")
        if confirm:
            del self.data["Right"][gesture_name]
            save_gestures(self.data)
            self.refresh_lists()

if __name__ == "__main__":
    app = GestureEditor()
    app.mainloop()
