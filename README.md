# Human Pose Estimation with MediaPipe and OpenCV

This project implements real-time human pose estimation using the MediaPipe library for pose detection and OpenCV for processing and displaying the results. The system detects human keypoints (such as joints and limbs) from an input video or live webcam feed and overlays the pose on the original image. Additionally, it displays the detected pose on a plain white background.

## Features
- Real-time pose detection on video or webcam input.
- Displays the detected pose on the original video feed.
- Shows the same detected pose on a blank white canvas for clarity.
- Prints the detected landmarks (keypoints) to the console.

## Requirements
- Python 3.9
- OpenCV
- MediaPipe
- NumPy

## Install Dependencies
You can install the required packages using pip:
```bash
pip install opencv-python mediapipe numpy
```

## Usage

### Input Source
The script can take a video file or webcam input for pose detection.
- Change the video file path in `cv2.VideoCapture("vid3.mp4")` to use your own video.
- Alternatively, uncomment the `cv2.VideoCapture(0)` line to use the live webcam feed.

### Running the Script
Simply run the Python script `pose_estimation.py`.

## Output
- The original video frame with pose landmarks will be displayed in a window titled "Pose Estimation."
- The same pose will be displayed on a blank white background in another window titled "Extracted Pose."

## Example Output
- **Pose on Original Image**: The detected pose is overlaid on the original video or webcam feed.
- **Pose on White Background**: The same pose is displayed on a blank white canvas for better visibility.

## Code Breakdown
- **Pose Detection**: Uses the `Pose` class from the MediaPipe library to detect and track human keypoints.
- **Drawing Pose**: `mp_draw.draw_landmarks()` draws the detected keypoints and their connections on the image.
- **Video Handling**: OpenCV is used to capture video frames and display the output in real time.
- **Console Output**: The detected landmarks are printed to the console, which contains the coordinates of each body joint.

## Future Work
- Optimize the system for real-time performance on mobile devices using TensorFlow Lite.
- Improve accuracy and handle occlusions by integrating multi-person pose detection.
- Fine-tune the model for specific use cases like fitness or rehabilitation.
