Self-Driving Car: End-to-End Behavioral Cloning

An autonomous driving system that predicts steering angles from camera
input using a convolutional neural network, combined with classical
computer vision lane detection and a proportional speed controller.


What It Does

This project demonstrates three layers of an autonomous driving stack:

  Perception  — Lane detection using Canny edge detection, Hough line
                transform, and region-of-interest masking

  Prediction  — A CNN trained via behavioral cloning that maps raw
                camera pixels directly to steering commands

  Control     — A P-controller that regulates speed based on a target
                speed limit, smoothly decelerating on approach

The model was trained on 10,395 images collected from the Udacity
self-driving simulator and deployed as a real-time SocketIO server
that responds to telemetry from the simulator.


Project Structure

  drive.py            Real-time inference server (SocketIO + Flask)
  lanes.py            Lane detection pipeline (OpenCV)
  model.h5            Trained CNN model weights (3 MB)
  driving_log.csv     Training data log — 3,465 entries
  test2.mp4           Sample video for lane detection testing
  training-images/    10,395 camera images (left, center, right)


How It Works

1. The Udacity simulator sends telemetry (camera image + speed)
   to drive.py via SocketIO

2. The image is preprocessed:
   - Cropped to remove sky and car hood
   - Converted to YUV color space
   - Gaussian blur for noise reduction
   - Resized to 200x66 pixels
   - Normalized to [0, 1]

3. The CNN predicts a steering angle from the processed image

4. A P-controller computes throttle:
   throttle = max(0.15, 1.0 - speed / 10.0)
   This smoothly reduces speed as it approaches the 10 mph limit

5. Lane detection (lanes.py) independently processes frames:
   Canny edges -> ROI mask -> Hough lines -> left/right lane
   separation by slope -> averaging for stability


CNN Architecture

  5 convolutional layers with ReLU activation
  3 fully connected layers
  Output: single steering angle (-1.0 to +1.0)
  Input: 66x200x3 YUV images
  Trained via behavioral cloning — the model learns to mimic
  human driving from recorded steering data


Training Data

  10,395 images from Udacity simulator
  3,465 driving log entries (center, left, right cameras)
  Steering range: -1.0 (full left) to +1.0 (full right)
  22% of samples contain significant steering (>0.1)


How To Run

  1. Install dependencies:
     pip install tensorflow flask-socketio eventlet opencv-python pillow

  2. Download the Udacity Self-Driving Car Simulator

  3. Clone this repo

  4. Start the simulator in Autonomous Mode

  5. Run the server:
     python drive.py

  6. The car will drive itself


Lane Detection Demo

  python lanes.py

  Processes test2.mp4 and shows real-time lane overlay with:
  - Canny edge detection
  - Hough line transform
  - Left/right lane classification
  - Averaged lane boundaries


Technologies

  Python, TensorFlow/Keras, OpenCV, Flask-SocketIO, Eventlet,
  NumPy, PIL, Udacity Self-Driving Car Simulator
