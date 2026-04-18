# Computer Vision Dashboard: Real-Time Edge Detection
This repository contains a Python-based real-time video processing application developed for the Computer Vision course in the Robotics Engineering program at Universidad Carlos III de Madrid (UC3M). The project implements a complete computer vision pipeline, from color space transformations to advanced edge detection, all managed through an interactive mosaic dashboard.
## Description
The application captures a live video stream and processes it through multiple filters and algorithms simultaneously. It utilizes a mosaic layout to display the original feed, various color spaces, live histograms, and different edge detection results in a single synchronized window
## Key Features
- Multi-Space Visualization: Real-time display of Original (BGR), Grayscale, and HSV color spaces.
- Live Histograms: Dynamic computation and rendering of histograms for all color spaces, including multi-channel (B, G, R) and HSV overlays
- Interactive Parameter Control: Integrated trackbars to adjust Gaussian blur kernels and Canny edge detection thresholds on the fly
- Edge Detection Comparison: Side-by-side comparison of manual Sobel implementation, Canny, and Laplacian of Gaussian (LoG) methods.
- Performance Monitoring: Real-time FPS (Frames Per Second) counter overlay to monitor processing efficiency.
## Objective
The primary goal of this assignment was to build an integrated visual tool to understand how different image processing operations affect live video data. Specific technical objectives included:
- Implementing a manual Sobel operator by creating $3 \times 3$ kernels, applying them via cv2.filter2D, and computing the gradient magnitude.
- Managing complex window layouts using image concatenation (cv2.hconcat and cv2.vconcat).
- Applying Gaussian filtering to stabilize edge detection in noisy environments.
## Requirements
To run this project, you need a Python environment with the following dependencies installed:
- Python 3
- OpenCV (opencv-python)
- NumPy
- A functional webcam is required to provide the live video stream.
## How to Execute the Code
```bash
python ASSIGNMENT1.py
```
