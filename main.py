#!/usr/bin/env python3
"""
Face Recognition System
This system can detect and identify faces in images and videos based on known reference faces.
"""

import argparse
import cv2
from face_recognition_system import FaceRecognitionSystem


def main():
    parser = argparse.ArgumentParser(description='Face Recognition System')
    parser.add_argument('--mode', type=str, choices=['image', 'video', 'webcam'], 
                       default='image', help='Recognition mode')
    parser.add_argument('--known-faces', type=str, required=True,
                       help='Path to folder containing known face images')
    parser.add_argument('--input', type=str, 
                       help='Path to input image or video file (not needed for webcam mode)')
    parser.add_argument('--output', type=str,
                       help='Path to save output image (optional)')
    parser.add_argument('--tolerance', type=float, default=0.6,
                       help='Face matching tolerance (lower is stricter, default: 0.6)')
    
    args = parser.parse_args()
    
    # Initialize the face recognition system
    print("=" * 60)
    print("Face Recognition System")
    print("=" * 60)
    
    frs = FaceRecognitionSystem()
    
    # Load known faces
    print(f"\nLoading known faces from: {args.known_faces}")
    frs.load_known_faces_from_folder(args.known_faces)
    
    if len(frs.known_face_encodings) == 0:
        print("\n✗ No faces loaded! Please add face images to the known faces folder.")
        return
    
    print(f"\n✓ Loaded {len(frs.known_face_encodings)} known face(s)")
    print("-" * 60)
    
    # Process based on mode
    if args.mode == 'image':
        if not args.input:
            print("Error: --input required for image mode")
            return
        
        print(f"\nProcessing image: {args.input}")
        result_image, results = frs.find_faces_in_image(args.input, args.tolerance)
        
        if result_image is not None:
            print(f"\nFound {len(results)} face(s):")
            for i, result in enumerate(results, 1):
                print(f"  {i}. {result['name']} - Confidence: {result['confidence']:.2%}")
            
            # Display the result
            cv2.imshow('Face Recognition Result', result_image)
            print("\nPress any key to close the window...")
            cv2.waitKey(0)
            cv2.destroyAllWindows()
            
            # Save if output path provided
            if args.output:
                cv2.imwrite(args.output, result_image)
                print(f"\n✓ Result saved to: {args.output}")
    
    elif args.mode == 'video':
        if not args.input:
            print("Error: --input required for video mode")
            return
        
        print(f"\nProcessing video: {args.input}")
        frs.find_faces_in_video(args.input, args.tolerance)
    
    elif args.mode == 'webcam':
        print("\nStarting webcam face recognition...")
        print("Press 'q' to quit")
        frs.find_faces_in_video(0, args.tolerance)
    
    print("\n" + "=" * 60)
    print("Face Recognition Complete!")
    print("=" * 60)


if __name__ == '__main__':
    main()
