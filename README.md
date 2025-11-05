# Face Recognition System

A Python-based face recognition system that detects and identifies faces in images, videos, and real-time webcam feeds based on reference face images.

## Features

- **Multiple Detection Modes**: Image, Video, and Webcam support
- **High Accuracy**: Uses state-of-the-art face recognition algorithms
- **Real-time Processing**: Fast face detection and recognition
- **Easy to Use**: Simple command-line interface
- **Visual Output**: Displays bounding boxes and confidence scores

## Requirements

- Python 3.7+
- Webcam (optional, for real-time recognition)

## Installation

1. **Install dependencies:**
```bash
pip install -r requirements.txt
```

Note: On Windows, you may need to install CMake and dlib separately:
```bash
pip install cmake
pip install dlib
```

2. **Prepare known faces:**
   - Create a folder (e.g., `known_faces`)
   - Add images of people you want to recognize
   - Name each image file with the person's name (e.g., `john_doe.jpg`, `jane_smith.png`)
   - Each image should contain only one face

## Project Structure

```
python_project_2/
├── face_recognition_system.py  # Core face recognition class
├── main.py                     # Main application entry point
├── requirements.txt            # Python dependencies
├── README.md                   # This file
├── known_faces/               # Folder for reference face images (create this)
│   ├── person1.jpg
│   ├── person2.jpg
│   └── ...
└── test_images/               # Folder for test images (optional)
    └── ...
```

## Usage

### 1. Image Recognition

Identify faces in a single image:

```bash
python main.py --mode image --known-faces ./known_faces --input test_image.jpg
```

Save the result to a file:

```bash
python main.py --mode image --known-faces ./known_faces --input test_image.jpg --output result.jpg
```

### 2. Video Recognition

Process a video file:

```bash
python main.py --mode video --known-faces ./known_faces --input test_video.mp4
```

### 3. Webcam Recognition (Real-time)

Use your webcam for real-time face recognition:

```bash
python main.py --mode webcam --known-faces ./known_faces
```

Press 'q' to quit the webcam view.

### 4. Adjust Tolerance

Control matching strictness (default: 0.6, lower = stricter):

```bash
python main.py --mode image --known-faces ./known_faces --input test.jpg --tolerance 0.5
```

## How It Works

1. **Loading Known Faces**: The system loads reference images from the `known_faces` folder and creates face encodings (mathematical representations of faces)

2. **Face Detection**: Uses HOG (Histogram of Oriented Gradients) or CNN to locate faces in the input

3. **Face Encoding**: Converts detected faces into 128-dimensional encodings

4. **Face Comparison**: Compares detected face encodings with known face encodings using Euclidean distance

5. **Identification**: Matches faces with confidence scores and displays results

## Tips for Best Results

- Use clear, well-lit photos for reference faces
- Include only one face per reference image
- Use front-facing photos for better accuracy
- Adjust tolerance parameter if getting too many false positives/negatives
- For video/webcam, ensure good lighting conditions

## Troubleshooting

**Issue**: "No face found in image"
- Ensure the image contains a clear, visible face
- Try a different image with better lighting
- Check that the image file is not corrupted

**Issue**: Slow performance
- Reduce video resolution
- Increase `process_every_n_frames` in the code
- Use a GPU-accelerated dlib build

**Issue**: Installation problems on Windows
- Install Visual C++ Build Tools
- Install CMake: `pip install cmake`
- Try: `pip install --upgrade pip setuptools wheel`

## License

This project is provided as-is for educational and personal use.

## Acknowledgments

- Built using the [face_recognition](https://github.com/ageitgey/face_recognition) library by Adam Geitgey
- Uses dlib's state-of-the-art face recognition models
