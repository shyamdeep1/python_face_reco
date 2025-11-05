#!/usr/bin/env python3
"""
Face Recognition System - GUI Application
A user-friendly graphical interface for the face recognition system
"""

import tkinter as tk
from tkinter import ttk, filedialog, messagebox, scrolledtext
import cv2
from PIL import Image, ImageTk
import threading
import os
from face_recognition_system import FaceRecognitionSystem


class FaceRecognitionGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("Face Recognition System - AI Powered")
        self.root.geometry("1200x800")
        self.root.resizable(True, True)
        
        # Set color scheme
        self.colors = {
            'primary': '#2C3E50',      # Dark blue-gray
            'secondary': '#3498DB',    # Bright blue
            'accent': '#E74C3C',       # Red
            'success': '#27AE60',      # Green
            'warning': '#F39C12',      # Orange
            'dark': '#1C2833',         # Very dark
            'light': '#ECF0F1',        # Light gray
            'white': '#FFFFFF',
            'text_dark': '#2C3E50',
            'text_light': '#7F8C8D'
        }
        
        # Configure root window
        self.root.configure(bg=self.colors['light'])
        
        # Initialize variables
        self.frs = FaceRecognitionSystem()
        self.known_faces_folder = tk.StringVar(value="./known_faces")
        self.tolerance = tk.DoubleVar(value=0.6)
        self.webcam_running = False
        self.video_thread = None
        self.cap = None
        
        # Configure styles
        self.setup_styles()
        
        # Create GUI
        self.create_widgets()
    
    def setup_styles(self):
        """Configure custom ttk styles"""
        style = ttk.Style()
        style.theme_use('clam')
        
        # Configure main frame
        style.configure('Main.TFrame', background=self.colors['light'])
        
        # Configure label frames
        style.configure('Card.TLabelframe', 
                       background=self.colors['white'],
                       borderwidth=2,
                       relief='flat')
        style.configure('Card.TLabelframe.Label', 
                       background=self.colors['white'],
                       foreground=self.colors['text_dark'],
                       font=('Segoe UI', 11, 'bold'))
        
        # Configure buttons
        style.configure('Primary.TButton',
                       background=self.colors['secondary'],
                       foreground=self.colors['white'],
                       borderwidth=0,
                       focuscolor='none',
                       padding=(15, 10),
                       font=('Segoe UI', 10, 'bold'))
        style.map('Primary.TButton',
                 background=[('active', '#2980B9'), ('pressed', '#21618C')])
        
        style.configure('Success.TButton',
                       background=self.colors['success'],
                       foreground=self.colors['white'],
                       borderwidth=0,
                       focuscolor='none',
                       padding=(15, 10),
                       font=('Segoe UI', 10, 'bold'))
        style.map('Success.TButton',
                 background=[('active', '#229954'), ('pressed', '#1E8449')])
        
        style.configure('Danger.TButton',
                       background=self.colors['accent'],
                       foreground=self.colors['white'],
                       borderwidth=0,
                       focuscolor='none',
                       padding=(15, 10),
                       font=('Segoe UI', 10, 'bold'))
        style.map('Danger.TButton',
                 background=[('active', '#C0392B'), ('pressed', '#A93226')])
        
        # Configure entries
        style.configure('Custom.TEntry',
                       fieldbackground=self.colors['white'],
                       borderwidth=2,
                       relief='flat')
        
        # Configure labels
        style.configure('Title.TLabel',
                       background=self.colors['light'],
                       foreground=self.colors['primary'],
                       font=('Segoe UI', 24, 'bold'))
        
        style.configure('Subtitle.TLabel',
                       background=self.colors['white'],
                       foreground=self.colors['text_light'],
                       font=('Segoe UI', 9))
        
        style.configure('Status.TLabel',
                       background=self.colors['white'],
                       foreground=self.colors['text_dark'],
                       font=('Segoe UI', 10, 'bold'))
        
    def create_widgets(self):
        # Main container
        main_frame = ttk.Frame(self.root, padding="15", style='Main.TFrame')
        main_frame.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        
        # Configure grid weights
        self.root.columnconfigure(0, weight=1)
        self.root.rowconfigure(0, weight=1)
        main_frame.columnconfigure(1, weight=1)
        main_frame.rowconfigure(4, weight=1)
        
        # Header Frame
        header_frame = tk.Frame(main_frame, bg=self.colors['primary'], height=80)
        header_frame.grid(row=0, column=0, columnspan=3, sticky=(tk.W, tk.E), pady=(0, 15))
        header_frame.grid_propagate(False)
        
        # Title with icon
        title_label = tk.Label(header_frame, 
                              text="🎭 Face Recognition System", 
                              font=('Segoe UI', 26, 'bold'),
                              bg=self.colors['primary'],
                              fg=self.colors['white'])
        title_label.pack(side=tk.TOP, pady=(15, 5))
        
        subtitle_label = tk.Label(header_frame,
                                 text="AI-Powered Face Detection & Recognition",
                                 font=('Segoe UI', 11),
                                 bg=self.colors['primary'],
                                 fg=self.colors['light'])
        subtitle_label.pack(side=tk.TOP)
        
        # Settings Frame
        settings_frame = ttk.LabelFrame(main_frame, text="⚙️  Configuration", 
                                       padding="15", style='Card.TLabelframe')
        settings_frame.grid(row=1, column=0, columnspan=3, sticky=(tk.W, tk.E), pady=(0, 10))
        settings_frame.columnconfigure(1, weight=1)
        
        # Known faces folder
        ttk.Label(settings_frame, text="Known Faces Folder:", 
                 style='Subtitle.TLabel').grid(row=0, column=0, sticky=tk.W, pady=8, padx=(0, 10))
        
        folder_frame = tk.Frame(settings_frame, bg=self.colors['white'])
        folder_frame.grid(row=0, column=1, columnspan=2, sticky=(tk.W, tk.E), padx=(0, 10))
        
        folder_entry = tk.Entry(folder_frame, textvariable=self.known_faces_folder, 
                               font=('Segoe UI', 10),
                               bg=self.colors['light'],
                               fg=self.colors['text_dark'],
                               relief='flat',
                               borderwidth=2)
        folder_entry.pack(side=tk.LEFT, fill=tk.X, expand=True, ipady=8, padx=(0, 5))
        
        ttk.Button(settings_frame, text="📁 Browse", 
                  command=self.browse_folder,
                  style='Primary.TButton').grid(row=0, column=3, padx=5)
        ttk.Button(settings_frame, text="✓ Load Faces", 
                  command=self.load_known_faces,
                  style='Success.TButton').grid(row=0, column=4, padx=5)
        
        # Tolerance slider
        ttk.Label(settings_frame, text="Match Tolerance:", 
                 style='Subtitle.TLabel').grid(row=1, column=0, sticky=tk.W, pady=8)
        
        tolerance_container = tk.Frame(settings_frame, bg=self.colors['white'])
        tolerance_container.grid(row=1, column=1, columnspan=2, sticky=(tk.W, tk.E), pady=8)
        
        tolerance_scale = ttk.Scale(tolerance_container, from_=0.3, to=1.0, 
                                   variable=self.tolerance, orient=tk.HORIZONTAL)
        tolerance_scale.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=(0, 10))
        
        self.tolerance_label = tk.Label(tolerance_container, 
                                       textvariable=self.tolerance,
                                       font=('Segoe UI', 10, 'bold'),
                                       bg=self.colors['secondary'],
                                       fg=self.colors['white'],
                                       padx=10, pady=5,
                                       relief='flat')
        self.tolerance_label.pack(side=tk.LEFT)
        
        # Add tolerance info
        info_label = ttk.Label(settings_frame, 
                              text="Lower = Stricter matching  |  Higher = More flexible",
                              style='Subtitle.TLabel')
        info_label.grid(row=1, column=3, columnspan=2, sticky=tk.W, padx=10)
        
        # Status label with icon
        status_container = tk.Frame(settings_frame, bg=self.colors['light'], relief='flat')
        status_container.grid(row=2, column=0, columnspan=5, pady=(10, 0), sticky=(tk.W, tk.E))
        
        self.status_label = tk.Label(status_container, 
                                     text="⚠️  No faces loaded - Please load known faces to begin",
                                     font=('Segoe UI', 11, 'bold'),
                                     bg=self.colors['light'],
                                     fg=self.colors['accent'],
                                     pady=10)
        self.status_label.pack(fill=tk.X)
        
        # Action Buttons Frame
        action_frame = ttk.LabelFrame(main_frame, text="🎬  Actions", 
                                     padding="15", style='Card.TLabelframe')
        action_frame.grid(row=2, column=0, columnspan=3, sticky=(tk.W, tk.E), pady=(0, 10))
        
        # Create buttons in a grid layout with better spacing
        button_container = tk.Frame(action_frame, bg=self.colors['white'])
        button_container.pack(fill=tk.X)
        
        self.btn_image = ttk.Button(button_container, text="📷  Analyze Image", 
                                    command=self.recognize_image, 
                                    style='Primary.TButton',
                                    width=18)
        self.btn_image.pack(side=tk.LEFT, padx=5, pady=5, expand=True, fill=tk.X)
        
        self.btn_video = ttk.Button(button_container, text="🎥  Process Video", 
                                    command=self.recognize_video,
                                    style='Primary.TButton',
                                    width=18)
        self.btn_video.pack(side=tk.LEFT, padx=5, pady=5, expand=True, fill=tk.X)
        
        self.btn_webcam = ttk.Button(button_container, text="📹  Start Webcam", 
                                     command=self.toggle_webcam,
                                     style='Success.TButton',
                                     width=18)
        self.webcam_button_text = "📹  Start Webcam"
        self.btn_webcam.pack(side=tk.LEFT, padx=5, pady=5, expand=True, fill=tk.X)
        
        self.btn_clear = ttk.Button(button_container, text="🗑️  Clear Display", 
                                    command=self.clear_display,
                                    style='Danger.TButton',
                                    width=18)
        self.btn_clear.pack(side=tk.LEFT, padx=5, pady=5, expand=True, fill=tk.X)
        
        # Display Frame (for image/video)
        display_frame = ttk.LabelFrame(main_frame, text="📺  Live Display", 
                                      padding="15", style='Card.TLabelframe')
        display_frame.grid(row=3, column=0, columnspan=2, sticky=(tk.W, tk.E, tk.N, tk.S), pady=(0, 10))
        display_frame.columnconfigure(0, weight=1)
        display_frame.rowconfigure(0, weight=1)
        
        # Canvas for displaying images with border
        canvas_container = tk.Frame(display_frame, bg=self.colors['dark'], relief='flat', borderwidth=3)
        canvas_container.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        canvas_container.grid_propagate(False)
        
        self.canvas = tk.Canvas(canvas_container, 
                               bg=self.colors['dark'], 
                               width=700, 
                               height=500,
                               highlightthickness=0)
        self.canvas.pack(fill=tk.BOTH, expand=True)
        
        # Add placeholder text
        self.canvas.create_text(350, 250, 
                               text="No image loaded\n\nSelect an action to begin",
                               font=('Segoe UI', 16),
                               fill=self.colors['text_light'],
                               tags='placeholder')
        
        # Results Frame
        results_frame = ttk.LabelFrame(main_frame, text="📊  Recognition Results", 
                                      padding="15", style='Card.TLabelframe')
        results_frame.grid(row=3, column=2, sticky=(tk.W, tk.E, tk.N, tk.S), pady=(0, 10), padx=(10, 0))
        results_frame.columnconfigure(0, weight=1)
        results_frame.rowconfigure(0, weight=1)
        
        # Results text area with custom styling
        self.results_text = scrolledtext.ScrolledText(results_frame, 
                                                      width=35, 
                                                      height=20, 
                                                      wrap=tk.WORD, 
                                                      font=('Consolas', 10),
                                                      bg=self.colors['light'],
                                                      fg=self.colors['text_dark'],
                                                      relief='flat',
                                                      borderwidth=2)
        self.results_text.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        
        # Log Frame
        log_frame = ttk.LabelFrame(main_frame, text="📝  Activity Log", 
                                  padding="15", style='Card.TLabelframe')
        log_frame.grid(row=4, column=0, columnspan=3, sticky=(tk.W, tk.E, tk.N, tk.S), pady=0)
        log_frame.columnconfigure(0, weight=1)
        log_frame.rowconfigure(0, weight=1)
        
        # Log text area with custom styling
        self.log_text = scrolledtext.ScrolledText(log_frame, 
                                                  height=7, 
                                                  wrap=tk.WORD, 
                                                  font=('Consolas', 9),
                                                  bg=self.colors['dark'],
                                                  fg=self.colors['success'],
                                                  relief='flat',
                                                  borderwidth=0,
                                                  insertbackground=self.colors['success'])
        self.log_text.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        
        # Initial log message
        self.log("✓ Face Recognition System initialized successfully")
        self.log("✓ Ready to process images, videos, and webcam feeds")
        self.log("⚡ Please load known faces to begin recognition")
        
    def log(self, message, level='info'):
        """Add message to log with color coding"""
        timestamp = self.get_timestamp()
        
        # Color code based on level
        if level == 'error':
            color_tag = 'error'
            icon = '✗'
        elif level == 'success':
            color_tag = 'success'
            icon = '✓'
        elif level == 'warning':
            color_tag = 'warning'
            icon = '⚠'
        else:
            color_tag = 'info'
            icon = '➤'
        
        # Configure tags
        self.log_text.tag_config('error', foreground='#E74C3C')
        self.log_text.tag_config('success', foreground='#27AE60')
        self.log_text.tag_config('warning', foreground='#F39C12')
        self.log_text.tag_config('info', foreground='#3498DB')
        self.log_text.tag_config('timestamp', foreground='#7F8C8D')
        
        self.log_text.insert(tk.END, f"[{timestamp}] ", 'timestamp')
        self.log_text.insert(tk.END, f"{icon} {message}\n", color_tag)
        self.log_text.see(tk.END)
        self.root.update_idletasks()
    
    def get_timestamp(self):
        """Get current timestamp"""
        from datetime import datetime
        return datetime.now().strftime("%H:%M:%S")
        
    def browse_folder(self):
        """Browse for known faces folder"""
        folder = filedialog.askdirectory(title="Select Known Faces Folder")
        if folder:
            self.known_faces_folder.set(folder)
            
    def load_known_faces(self):
        """Load known faces from folder"""
        folder = self.known_faces_folder.get()
        if not os.path.exists(folder):
            messagebox.showerror("Error", f"Folder not found: {folder}")
            self.log(f"Folder not found: {folder}", 'error')
            return
            
        self.log(f"Loading faces from: {folder}")
        try:
            self.frs = FaceRecognitionSystem()
            self.frs.load_known_faces_from_folder(folder)
            
            if len(self.frs.known_face_encodings) == 0:
                self.status_label.config(
                    text="⚠️  No faces found in folder - Please add face images",
                    bg=self.colors['warning'],
                    fg=self.colors['white'])
                self.log("No valid face images found in folder", 'warning')
                messagebox.showwarning("Warning", "No face images found in the selected folder.\n\n"
                                      "Please add images with clear faces (one face per image).")
            else:
                count = len(self.frs.known_face_encodings)
                self.status_label.config(
                    text=f"✓  {count} face(s) loaded successfully - System ready!",
                    bg=self.colors['success'],
                    fg=self.colors['white'])
                self.log(f"Successfully loaded {count} face(s)", 'success')
                self.log(f"Loaded: {', '.join(self.frs.known_face_names)}", 'success')
        except Exception as e:
            self.log(f"Error loading faces: {str(e)}", 'error')
            messagebox.showerror("Error", f"Failed to load faces:\n{str(e)}")
            
    def recognize_image(self):
        """Recognize faces in an image file"""
        if len(self.frs.known_face_encodings) == 0:
            messagebox.showwarning("Warning", "Please load known faces first!")
            self.log("Cannot process: No known faces loaded", 'warning')
            return
            
        file_path = filedialog.askopenfilename(
            title="Select Image",
            filetypes=[("Image files", "*.jpg *.jpeg *.png *.bmp"), ("All files", "*.*")]
        )
        
        if not file_path:
            return
            
        self.log(f"Processing image: {os.path.basename(file_path)}")
        self.results_text.delete(1.0, tk.END)
        self.results_text.insert(tk.END, "Processing...\nPlease wait...\n", 'info')
        self.root.update_idletasks()
        
        try:
            result_image, results = self.frs.find_faces_in_image(file_path, self.tolerance.get())
            
            if result_image is not None:
                # Clear placeholder
                self.canvas.delete('placeholder')
                
                # Display image
                self.display_image(result_image)
                
                # Show results with better formatting
                self.results_text.delete(1.0, tk.END)
                self.results_text.insert(tk.END, "═" * 40 + "\n", 'header')
                self.results_text.insert(tk.END, f"   DETECTED FACES: {len(results)}\n", 'header')
                self.results_text.insert(tk.END, "═" * 40 + "\n\n", 'header')
                
                for i, result in enumerate(results, 1):
                    self.results_text.insert(tk.END, f"Person #{i}\n", 'bold')
                    self.results_text.insert(tk.END, f"  Name: {result['name']}\n")
                    self.results_text.insert(tk.END, f"  Confidence: {result['confidence']:.2%}\n")
                    self.results_text.insert(tk.END, f"  Position: {result['location']}\n\n")
                
                self.results_text.tag_config('header', font=('Consolas', 10, 'bold'))
                self.results_text.tag_config('bold', font=('Consolas', 10, 'bold'))
                
                self.log(f"Successfully detected {len(results)} face(s)", 'success')
            else:
                self.results_text.delete(1.0, tk.END)
                self.results_text.insert(tk.END, "No faces detected in image")
                self.log("No faces detected in image", 'warning')
                messagebox.showinfo("Info", "No faces detected in the image")
        except Exception as e:
            self.log(f"Error: {str(e)}", 'error')
            messagebox.showerror("Error", f"Failed to process image:\n{str(e)}")
            
    def recognize_video(self):
        """Recognize faces in a video file"""
        if len(self.frs.known_face_encodings) == 0:
            messagebox.showwarning("Warning", "Please load known faces first!")
            self.log("Cannot process video: No known faces loaded", 'warning')
            return
            
        file_path = filedialog.askopenfilename(
            title="Select Video",
            filetypes=[("Video files", "*.mp4 *.avi *.mov *.mkv"), ("All files", "*.*")]
        )
        
        if not file_path:
            return
            
        self.log(f"Processing video: {os.path.basename(file_path)}", 'info')
        
        # Ask user for display preference
        choice = messagebox.askyesnocancel(
            "Video Display Options",
            "Do you want to display video in a SEPARATE window?\n\n"
            "• YES - Opens in new resizable window (can loop)\n"
            "• NO - Display in GUI canvas (one-time playback)\n"
            "• CANCEL - Go back\n\n"
            "Press 'Q' to stop video playback anytime."
        )
        
        if choice is None:  # Cancel
            return
        elif choice:  # Yes - separate window
            self.log("Opening video in separate window - Press 'Q' to close", 'info')
            
            try:
                # Run in separate thread to avoid blocking GUI
                def process_video():
                    try:
                        self.frs.find_faces_in_video(file_path, self.tolerance.get())
                        self.log("Video window closed", 'success')
                    except Exception as e:
                        self.log(f"Video error: {str(e)}", 'error')
                    
                thread = threading.Thread(target=process_video, daemon=True)
                thread.start()
            except Exception as e:
                self.log(f"Error: {str(e)}", 'error')
                messagebox.showerror("Error", f"Failed to process video:\n{str(e)}")
        else:  # No - display in GUI
            self.log("Playing video in GUI canvas", 'info')
            self.play_video_in_gui(file_path)
            
    def toggle_webcam(self):
        """Start or stop webcam"""
        if not self.webcam_running:
            self.start_webcam()
        else:
            self.stop_webcam()
            
    def start_webcam(self):
        """Start webcam face recognition"""
        if len(self.frs.known_face_encodings) == 0:
            messagebox.showwarning("Warning", "Please load known faces first!")
            self.log("Cannot start webcam: No known faces loaded", 'warning')
            return
        
        # Clear placeholder
        self.canvas.delete('placeholder')
        
        self.webcam_running = True
        self.btn_webcam.config(text="⏹  Stop Webcam")
        self.log("Starting webcam...", 'success')
        self.results_text.delete(1.0, tk.END)
        self.results_text.insert(tk.END, "🔴 WEBCAM ACTIVE\n", 'header')
        self.results_text.insert(tk.END, "═" * 40 + "\n\n", 'header')
        self.results_text.tag_config('header', font=('Consolas', 11, 'bold'), foreground='#E74C3C')
        
        def webcam_loop():
            self.cap = cv2.VideoCapture(0)
            
            if not self.cap.isOpened():
                self.log("Error: Could not open webcam", 'error')
                self.webcam_running = False
                self.btn_webcam.config(text="📹  Start Webcam")
                return
            
            self.log("Webcam connected successfully", 'success')
                
            while self.webcam_running:
                ret, frame = self.cap.read()
                if not ret:
                    break
                    
                # Process frame
                result_frame, results = self.frs.process_frame(frame, self.tolerance.get())
                
                # Display frame
                self.display_image(result_frame)
                
                # Update results (limit updates to avoid flickering)
                if len(results) > 0:
                    self.results_text.delete(1.0, tk.END)
                    self.results_text.insert(tk.END, "Detected Faces:\n")
                    self.results_text.insert(tk.END, "=" * 30 + "\n\n")
                    for i, result in enumerate(results, 1):
                        self.results_text.insert(tk.END, f"{i}. {result['name']}\n")
                        self.results_text.insert(tk.END, f"   Conf: {result['confidence']:.1%}\n\n")
                
            self.cap.release()
            self.log("Webcam stopped")
            
        self.video_thread = threading.Thread(target=webcam_loop, daemon=True)
        self.video_thread.start()
        
    def stop_webcam(self):
        """Stop webcam"""
        self.webcam_running = False
        self.btn_webcam.config(text="📹  Start Webcam")
        self.log("Webcam stopped", 'info')
    
    def play_video_in_gui(self, video_path):
        """Play video directly in the GUI canvas"""
        self.canvas.delete('placeholder')
        
        def video_player():
            cap = cv2.VideoCapture(video_path)
            if not cap.isOpened():
                self.log("Failed to open video file", 'error')
                return
            
            self.log("Playing video in canvas...", 'info')
            
            try:
                while cap.isOpened():
                    ret, frame = cap.read()
                    if not ret:
                        self.log("Video playback finished", 'success')
                        break
                    
                    # Process frame with face recognition
                    result_frame, results = self.frs.process_frame(frame, self.tolerance.get())
                    
                    # Display in canvas
                    self.display_image(result_frame)
                    
                    # Update results if faces found
                    if len(results) > 0:
                        self.results_text.delete(1.0, tk.END)
                        self.results_text.insert(tk.END, "Detected Faces:\n", 'header')
                        self.results_text.insert(tk.END, "═" * 40 + "\n\n", 'header')
                        for i, result in enumerate(results, 1):
                            self.results_text.insert(tk.END, f"{i}. {result['name']}\n")
                            self.results_text.insert(tk.END, f"   Conf: {result['confidence']:.1%}\n\n")
                    
                    # Small delay for smoother playback
                    self.root.update()
                    cv2.waitKey(30)
                    
            except Exception as e:
                self.log(f"Video playback error: {str(e)}", 'error')
            finally:
                cap.release()
        
        # Run in separate thread
        thread = threading.Thread(target=video_player, daemon=True)
        thread.start()
        
    def display_image(self, cv_image):
        """Display OpenCV image on canvas"""
        # Convert BGR to RGB
        rgb_image = cv2.cvtColor(cv_image, cv2.COLOR_BGR2RGB)
        
        # Resize to fit canvas
        canvas_width = self.canvas.winfo_width()
        canvas_height = self.canvas.winfo_height()
        
        if canvas_width > 1 and canvas_height > 1:
            h, w = rgb_image.shape[:2]
            scale = min(canvas_width/w, canvas_height/h)
            new_w, new_h = int(w*scale), int(h*scale)
            rgb_image = cv2.resize(rgb_image, (new_w, new_h))
        
        # Convert to PIL Image
        pil_image = Image.fromarray(rgb_image)
        
        # Convert to PhotoImage
        photo = ImageTk.PhotoImage(image=pil_image)
        
        # Display on canvas
        self.canvas.delete("all")
        self.canvas.create_image(canvas_width//2, canvas_height//2, image=photo, anchor=tk.CENTER)
        self.canvas.image = photo  # Keep reference
        
    def clear_display(self):
        """Clear canvas and results"""
        self.canvas.delete("all")
        self.results_text.delete(1.0, tk.END)
        self.log("Display cleared")
        
    def on_closing(self):
        """Handle window closing"""
        if self.webcam_running:
            self.stop_webcam()
        self.root.destroy()


def main():
    root = tk.Tk()
    app = FaceRecognitionGUI(root)
    root.protocol("WM_DELETE_WINDOW", app.on_closing)
    root.mainloop()


if __name__ == '__main__':
    main()
