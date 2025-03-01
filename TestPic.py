import tkinter as tk
from tkinter import filedialog, messagebox, ttk
import os
import cv2
import numpy as np
from PIL import Image, ImageTk
from ultralytics import YOLO
import torch

class LeakDetectionTester:
    def __init__(self, root):
        self.root = root
        self.root.title("Leak Detection Testing Tool")
        self.root.geometry("1200x800")
        
        # Variables
        self.model_path = None
        self.model = None
        self.image_path = None
        self.original_image = None
        self.detection_image = None
        self.confidence_threshold = 0.25
        
        # Create UI components
        self.create_menu()
        self.create_main_frame()
        
        # Status bar
        self.status_var = tk.StringVar()
        self.status_var.set("Ready. Please load model and image.")
        self.status_bar = tk.Label(root, textvariable=self.status_var, bd=1, relief=tk.SUNKEN, anchor=tk.W)
        self.status_bar.pack(side=tk.BOTTOM, fill=tk.X)
        
        # Check if CUDA is available
        self.cuda_available = torch.cuda.is_available()
        if self.cuda_available:
            self.status_var.set("CUDA available. GPU will be used for detection.")
        else:
            self.status_var.set("CUDA not available. CPU will be used for detection.")
    
    def create_menu(self):
        menubar = tk.Menu(self.root)
        
        file_menu = tk.Menu(menubar, tearoff=0)
        file_menu.add_command(label="Load Model", command=self.load_model)
        file_menu.add_command(label="Open Image", command=self.open_image)
        file_menu.add_separator()
        file_menu.add_command(label="Exit", command=self.root.quit)
        menubar.add_cascade(label="File", menu=file_menu)
        
        self.root.config(menu=menubar)
    
    def create_main_frame(self):
        # Main frame
        main_frame = tk.Frame(self.root)
        main_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        # Left panel (controls)
        control_frame = tk.Frame(main_frame, width=200)
        control_frame.pack(side=tk.LEFT, fill=tk.Y, padx=(0, 10))
        
        # Model frame
        model_frame = tk.LabelFrame(control_frame, text="Model", padx=5, pady=5)
        model_frame.pack(fill=tk.X, pady=(0, 10))
        
        load_model_btn = tk.Button(model_frame, text="Load Model", command=self.load_model)
        load_model_btn.pack(fill=tk.X, pady=5)
        
        self.model_label = tk.Label(model_frame, text="No model loaded", wraplength=180)
        self.model_label.pack(fill=tk.X, pady=5)
        
        # Image frame
        image_frame = tk.LabelFrame(control_frame, text="Image", padx=5, pady=5)
        image_frame.pack(fill=tk.X, pady=(0, 10))
        
        load_image_btn = tk.Button(image_frame, text="Load Image", command=self.open_image)
        load_image_btn.pack(fill=tk.X, pady=5)
        
        # Detection frame
        detection_frame = tk.LabelFrame(control_frame, text="Detection", padx=5, pady=5)
        detection_frame.pack(fill=tk.X, pady=(0, 10))
        
        detect_btn = tk.Button(detection_frame, text="Detect Leaks", command=self.detect_leaks)
        detect_btn.pack(fill=tk.X, pady=5)
        
        # Confidence threshold
        threshold_frame = tk.Frame(detection_frame)
        threshold_frame.pack(fill=tk.X, pady=5)
        
        tk.Label(threshold_frame, text="Confidence:").pack(side=tk.LEFT)
        
        self.threshold_var = tk.DoubleVar(value=0.25)
        threshold_scale = ttk.Scale(
            threshold_frame, 
            from_=0.1, 
            to=1.0, 
            orient='horizontal',
            variable=self.threshold_var,
            command=self.update_threshold
        )
        threshold_scale.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=5)
        
        self.threshold_label = tk.Label(threshold_frame, text="0.25")
        self.threshold_label.pack(side=tk.LEFT, padx=(0, 5))
        
        # Results frame
        results_frame = tk.LabelFrame(control_frame, text="Results", padx=5, pady=5)
        results_frame.pack(fill=tk.BOTH, expand=True)
        
        self.results_text = tk.Text(results_frame, wrap=tk.WORD, height=10)
        self.results_text.pack(fill=tk.BOTH, expand=True)
        
        # Right panel (image display)
        display_frame = tk.Frame(main_frame)
        display_frame.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True)
        
        # Image canvas
        self.canvas = tk.Canvas(display_frame, bg="gray")
        self.canvas.pack(fill=tk.BOTH, expand=True)
    
    def load_model(self):
        model_path = filedialog.askopenfilename(
            title="Select Model File",
            filetypes=(("PyTorch Model", "*.pt"), ("All Files", "*.*"))
        )
        
        if not model_path:
            return
        
        try:
            self.status_var.set("Loading model...")
            self.root.update()
            
            self.model = YOLO(model_path)
            self.model_path = model_path
            
            # Update UI
            model_name = os.path.basename(model_path)
            self.model_label.config(text=f"Model: {model_name}")
            self.status_var.set(f"Model loaded: {model_name}")
            
            # Clear results
            self.results_text.delete(1.0, tk.END)
            self.results_text.insert(tk.END, "Model loaded successfully.\n")
            self.results_text.insert(tk.END, f"Path: {model_path}\n")
            
        except Exception as e:
            messagebox.showerror("Error", f"Failed to load model: {str(e)}")
            self.status_var.set("Error loading model.")
    
    def open_image(self):
        image_path = filedialog.askopenfilename(
            title="Select Image",
            filetypes=(
                ("Image Files", "*.jpg *.jpeg *.png *.bmp"),
                ("JPEG", "*.jpg *.jpeg"),
                ("PNG", "*.png"),
                ("BMP", "*.bmp"),
                ("All Files", "*.*")
            )
        )
        
        if not image_path:
            return
        
        try:
            self.image_path = image_path
            self.original_image = cv2.imread(image_path)
            self.original_image = cv2.cvtColor(self.original_image, cv2.COLOR_BGR2RGB)
            
            # Display image
            self.display_image(self.original_image)
            
            # Update status
            img_name = os.path.basename(image_path)
            self.status_var.set(f"Image loaded: {img_name}")
            
            # Update results
            self.results_text.delete(1.0, tk.END)
            self.results_text.insert(tk.END, f"Image loaded: {img_name}\n")
            h, w = self.original_image.shape[:2]
            self.results_text.insert(tk.END, f"Dimensions: {w}x{h}\n")
            self.results_text.insert(tk.END, "Ready for detection.\n")
            
        except Exception as e:
            messagebox.showerror("Error", f"Failed to load image: {str(e)}")
            self.status_var.set("Error loading image.")
    
    def display_image(self, img, detections=False):
        # Get canvas dimensions
        canvas_width = self.canvas.winfo_width()
        canvas_height = self.canvas.winfo_height()
        
        # If canvas is not initialized yet, use default size
        if canvas_width <= 1:
            canvas_width = 800
        if canvas_height <= 1:
            canvas_height = 600
        
        # Get image dimensions
        img_height, img_width = img.shape[:2]
        
        # Calculate scale factor to fit image in canvas
        width_scale = canvas_width / img_width
        height_scale = canvas_height / img_height
        scale = min(width_scale, height_scale)
        
        # Resize image for display
        if scale < 1:
            new_width = int(img_width * scale)
            new_height = int(img_height * scale)
            img_display = cv2.resize(img, (new_width, new_height))
        else:
            img_display = img.copy()
        
        # Convert to PhotoImage
        img_pil = Image.fromarray(img_display)
        img_tk = ImageTk.PhotoImage(img_pil)
        
        # Display on canvas
        self.canvas.delete("all")
        self.canvas.create_image(canvas_width/2, canvas_height/2, anchor=tk.CENTER, image=img_tk)
        self.canvas.image = img_tk  # Keep a reference to prevent garbage collection
        
        # Update status
        if detections:
            self.status_var.set(f"Detection results displayed.")
        else:
            self.status_var.set(f"Image displayed.")
    
    def update_threshold(self, val):
        # Update threshold value
        self.confidence_threshold = self.threshold_var.get()
        self.threshold_label.config(text=f"{self.confidence_threshold:.2f}")
        
        # If we already have detection results, re-display with new threshold
        if hasattr(self, 'detection_results') and self.detection_results is not None:
            self.display_detection_results()
    
    def detect_leaks(self):
        if self.model is None:
            messagebox.showwarning("Warning", "Please load a model first.")
            return
        
        if self.original_image is None:
            messagebox.showwarning("Warning", "Please load an image first.")
            return
        
        try:
            self.status_var.set("Detecting leaks...")
            self.root.update()
            
            # Run inference
            results = self.model(self.original_image)
            self.detection_results = results[0]  # Store results for re-display with different thresholds
            
            # Display results
            self.display_detection_results()
            
        except Exception as e:
            messagebox.showerror("Error", f"Detection failed: {str(e)}")
            self.status_var.set("Error during detection.")
    
    def display_detection_results(self):
        # Create a copy of the original image
        img_with_boxes = self.original_image.copy()
        
        # Get detection results
        boxes = self.detection_results.boxes
        
        # Clear results text
        self.results_text.delete(1.0, tk.END)
        self.results_text.insert(tk.END, f"Detection Results:\n")
        
        # Filter by confidence threshold
        filtered_boxes = [box for box in boxes if box.conf.item() >= self.confidence_threshold]
        
        # Display number of detections
        self.results_text.insert(tk.END, f"Found {len(filtered_boxes)} leaks with confidence >= {self.confidence_threshold:.2f}\n\n")
        
        # Draw bounding boxes
        for i, box in enumerate(filtered_boxes):
            # Get box coordinates
            x1, y1, x2, y2 = box.xyxy[0].tolist()
            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
            
            # Get confidence
            conf = box.conf.item()
            
            # Draw box on image
            cv2.rectangle(img_with_boxes, (x1, y1), (x2, y2), (0, 255, 0), 2)
            
            # Add label
            label = f"Leak: {conf:.2f}"
            cv2.putText(img_with_boxes, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 3)
            
            # Add to results text
            self.results_text.insert(tk.END, f"Leak {i+1}:\n")
            self.results_text.insert(tk.END, f"  Position: ({x1}, {y1}) to ({x2}, {y2})\n")
            self.results_text.insert(tk.END, f"  Confidence: {conf:.4f}\n")
            
            # Calculate size
            width = x2 - x1
            height = y2 - y1
            self.results_text.insert(tk.END, f"  Size: {width}x{height} pixels\n\n")
        
        # Display image with boxes
        self.display_image(img_with_boxes, detections=True)

def main():
    root = tk.Tk()
    app = LeakDetectionTester(root)
    root.mainloop()

if __name__ == "__main__":
    main()