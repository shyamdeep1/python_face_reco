#!/usr/bin/env python3
"""
Example usage of the Face Recognition System as a library
"""

from face_recognition_system import FaceRecognitionSystem


def example_basic_usage():
    """Basic example: Load faces and recognize them in an image"""
    print("Example 1: Basic Face Recognition")
    print("-" * 40)
    
    # Create face recognition system
    frs = FaceRecognitionSystem()
    
    # Load a known face
    frs.load_known_face('path/to/reference_face.jpg', 'Person Name')
    
    # Or load all faces from a folder
    # frs.load_known_faces_from_folder('known_faces')
    
    # Find faces in a test image
    result_image, results = frs.find_faces_in_image('path/to/test_image.jpg')
    
    # Print results
    print(f"Found {len(results)} face(s):")
    for result in results:
        print(f"  - {result['name']} (Confidence: {result['confidence']:.2%})")
        print(f"    Location: {result['location']}")


def example_multiple_faces():
    """Example with multiple known faces"""
    print("\nExample 2: Multiple Known Faces")
    print("-" * 40)
    
    frs = FaceRecognitionSystem()
    
    # Load multiple known faces
    faces = [
        ('alice.jpg', 'Alice'),
        ('bob.jpg', 'Bob'),
        ('charlie.jpg', 'Charlie')
    ]
    
    for image_path, name in faces:
        frs.load_known_face(image_path, name)
    
    # Process an image
    result_image, results = frs.find_faces_in_image('group_photo.jpg')
    
    print(f"Recognized {len(results)} people in the group photo")


def example_webcam():
    """Example: Real-time webcam recognition"""
    print("\nExample 3: Webcam Recognition")
    print("-" * 40)
    
    frs = FaceRecognitionSystem()
    frs.load_known_faces_from_folder('known_faces')
    
    print("Starting webcam... Press 'q' to quit")
    frs.find_faces_in_video(0)  # 0 = default webcam


def example_custom_tolerance():
    """Example with custom tolerance for stricter matching"""
    print("\nExample 4: Custom Tolerance")
    print("-" * 40)
    
    frs = FaceRecognitionSystem()
    frs.load_known_faces_from_folder('known_faces')
    
    # Lower tolerance = stricter matching (less false positives)
    # Higher tolerance = looser matching (more false positives)
    
    print("Using strict tolerance (0.4):")
    result_image, results = frs.find_faces_in_image('test.jpg', tolerance=0.4)
    
    print(f"Found {len(results)} matches with strict settings")


if __name__ == '__main__':
    print("=" * 60)
    print("Face Recognition System - Example Usage")
    print("=" * 60)
    
    # Uncomment the example you want to run:
    
    # example_basic_usage()
    # example_multiple_faces()
    # example_webcam()
    # example_custom_tolerance()
    
    print("\nNote: Update file paths in the examples before running!")
    print("These are template examples showing how to use the library.")
