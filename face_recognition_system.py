import face_recognition
import cv2
import numpy as np
import os
from pathlib import Path


class FaceRecognitionSystem:
    def __init__(self):
        self.known_face_encodings = []
        self.known_face_names = []
        
    def load_known_face(self, image_path, person_name):
        """Load a known face from an image file"""
        try:
            image = face_recognition.load_image_file(image_path)
            face_encodings = face_recognition.face_encodings(image)
            
            if len(face_encodings) > 0:
                self.known_face_encodings.append(face_encodings[0])
                self.known_face_names.append(person_name)
                print(f"✓ Loaded face for: {person_name}")
                return True
            else:
                print(f"✗ No face found in: {image_path}")
                return False
        except Exception as e:
            print(f"✗ Error loading {image_path}: {str(e)}")
            return False
    
    def load_known_faces_from_folder(self, folder_path):
        """Load all faces from a folder. Each image filename becomes the person's name"""
        folder = Path(folder_path)
        if not folder.exists():
            print(f"Folder not found: {folder_path}")
            return
        
        image_extensions = ['.jpg', '.jpeg', '.png', '.bmp']
        for image_file in folder.iterdir():
            if image_file.suffix.lower() in image_extensions:
                person_name = image_file.stem
                self.load_known_face(str(image_file), person_name)
    
    def find_faces_in_image(self, image_path, tolerance=0.6):
        """Find and identify faces in an image"""
        try:
            image = face_recognition.load_image_file(image_path)
            face_locations = face_recognition.face_locations(image)
            face_encodings = face_recognition.face_encodings(image, face_locations)
            
            image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            
            results = []
            for (top, right, bottom, left), face_encoding in zip(face_locations, face_encodings):
                matches = face_recognition.compare_faces(
                    self.known_face_encodings, face_encoding, tolerance=tolerance
                )
                name = "Unknown"
                confidence = 0
                
                if True in matches:
                    face_distances = face_recognition.face_distance(
                        self.known_face_encodings, face_encoding
                    )
                    best_match_index = np.argmin(face_distances)
                    
                    if matches[best_match_index]:
                        name = self.known_face_names[best_match_index]
                        confidence = 1 - face_distances[best_match_index]
                
                results.append({
                    'name': name,
                    'confidence': confidence,
                    'location': (top, right, bottom, left)
                })
                
                # Draw rectangle and label
                cv2.rectangle(image_rgb, (left, top), (right, bottom), (0, 255, 0), 2)
                label = f"{name} ({confidence:.2%})" if name != "Unknown" else name
                cv2.rectangle(image_rgb, (left, bottom - 35), (right, bottom), (0, 255, 0), cv2.FILLED)
                cv2.putText(image_rgb, label, (left + 6, bottom - 6),
                           cv2.FONT_HERSHEY_DUPLEX, 0.6, (255, 255, 255), 1)
            
            return image_rgb, results
            
        except Exception as e:
            print(f"Error processing image: {str(e)}")
            return None, []
    
    def find_faces_in_video(self, video_source=0, tolerance=0.6):
        """Find and identify faces in real-time video"""
        video_capture = cv2.VideoCapture(video_source)
        
        if not video_capture.isOpened():
            print("Error: Could not open video source")
            return
        
        print("Press 'q' to quit video recognition")
        
        # Get video properties
        is_webcam = isinstance(video_source, int)
        total_frames = 0
        if not is_webcam:
            fps = int(video_capture.get(cv2.CAP_PROP_FPS))
            total_frames = int(video_capture.get(cv2.CAP_PROP_FRAME_COUNT))
            print(f"Video FPS: {fps}, Total frames: {total_frames}")
        
        process_every_n_frames = 2
        frame_count = 0
        loop_count = 0
        
        # Create window with specific properties
        window_name = 'Face Recognition - Press Q to quit'
        cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(window_name, 1280, 720)
        
        print("Video window opened. Press 'Q' to close.")
        
        try:
            while True:
                ret, frame = video_capture.read()
                
                # If video ended (not webcam), loop it
                if not ret:
                    if not is_webcam and total_frames > 0:
                        # Loop video - reset to beginning
                        loop_count += 1
                        print(f"Video ended - Looping (Loop #{loop_count}). Press 'Q' to quit.")
                        video_capture.set(cv2.CAP_PROP_POS_FRAMES, 0)
                        frame_count = 0
                        ret, frame = video_capture.read()
                        if not ret:
                            print("Error: Cannot read video after reset")
                            break
                    else:
                        print("Video/Webcam ended")
                        break
                
                frame_count += 1
                
                # Process faces
                if frame_count % process_every_n_frames == 0:
                    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    face_locations = face_recognition.face_locations(rgb_frame)
                    face_encodings = face_recognition.face_encodings(rgb_frame, face_locations)
                    
                    for (top, right, bottom, left), face_encoding in zip(face_locations, face_encodings):
                        matches = face_recognition.compare_faces(
                            self.known_face_encodings, face_encoding, tolerance=tolerance
                        )
                        name = "Unknown"
                        confidence = 0
                        
                        if True in matches:
                            face_distances = face_recognition.face_distance(
                                self.known_face_encodings, face_encoding
                            )
                            best_match_index = np.argmin(face_distances)
                            
                            if matches[best_match_index]:
                                name = self.known_face_names[best_match_index]
                                confidence = 1 - face_distances[best_match_index]
                        
                        # Draw rectangle and label
                        cv2.rectangle(frame, (left, top), (right, bottom), (0, 255, 0), 2)
                        label = f"{name} ({confidence:.2%})" if name != "Unknown" else name
                        cv2.rectangle(frame, (left, bottom - 35), (right, bottom), (0, 255, 0), cv2.FILLED)
                        cv2.putText(frame, label, (left + 6, bottom - 6),
                                   cv2.FONT_HERSHEY_DUPLEX, 0.6, (255, 255, 255), 1)
                
                # Add instructions on the frame
                instruction_text = "Press 'Q' to quit"
                cv2.putText(frame, instruction_text, (10, 30),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
                
                # Show frame info for videos
                if not is_webcam and total_frames > 0:
                    frame_info = f"Frame: {frame_count}/{total_frames} | Loop: {loop_count}"
                    cv2.putText(frame, frame_info, (10, 60),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 1)
                
                # Display the frame
                cv2.imshow(window_name, frame)
                
                # Check for key press - wait longer for better video playback
                key = cv2.waitKey(30 if not is_webcam else 1) & 0xFF
                if key == ord('q') or key == ord('Q') or key == 27:  # 27 is ESC
                    print("User pressed quit key")
                    break
                
                # Check if window was closed
                if cv2.getWindowProperty(window_name, cv2.WND_PROP_VISIBLE) < 1:
                    print("Window was closed by user")
                    break
                    
        except Exception as e:
            print(f"Error during video processing: {str(e)}")
        finally:
            # Cleanup
            video_capture.release()
            cv2.destroyAllWindows()
            cv2.waitKey(1)
            print("Video window closed and resources released.")
    
    def process_frame(self, frame, tolerance=0.6):
        """Process a single frame and return annotated frame with results"""
        try:
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            face_locations = face_recognition.face_locations(rgb_frame)
            face_encodings = face_recognition.face_encodings(rgb_frame, face_locations)
            
            results = []
            for (top, right, bottom, left), face_encoding in zip(face_locations, face_encodings):
                matches = face_recognition.compare_faces(
                    self.known_face_encodings, face_encoding, tolerance=tolerance
                )
                name = "Unknown"
                confidence = 0
                
                if True in matches:
                    face_distances = face_recognition.face_distance(
                        self.known_face_encodings, face_encoding
                    )
                    best_match_index = np.argmin(face_distances)
                    
                    if matches[best_match_index]:
                        name = self.known_face_names[best_match_index]
                        confidence = 1 - face_distances[best_match_index]
                
                results.append({
                    'name': name,
                    'confidence': confidence,
                    'location': (top, right, bottom, left)
                })
                
                # Draw rectangle and label
                cv2.rectangle(frame, (left, top), (right, bottom), (0, 255, 0), 2)
                label = f"{name} ({confidence:.2%})" if name != "Unknown" else name
                cv2.rectangle(frame, (left, bottom - 35), (right, bottom), (0, 255, 0), cv2.FILLED)
                cv2.putText(frame, label, (left + 6, bottom - 6),
                           cv2.FONT_HERSHEY_DUPLEX, 0.6, (255, 255, 255), 1)
            
            return frame, results
            
        except Exception as e:
            print(f"Error processing frame: {str(e)}")
            return frame, []
