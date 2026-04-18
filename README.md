# Feature Matching and Image Stitching Dashboard
This repository contains a Python-based application developed for the Computer Vision course in the Robotics Engineering program at Universidad Carlos III de Madrid (UC3M). The project focuses on advanced image processing techniques, specifically local feature detection, matching, and the creation of panoramic views through image stitching.
## Description
The application provides an environment to explore how different local feature descriptors and matching algorithms perform on static images. It processes a sequence of images to perform two main tasks: feature matching between pairs and the construction of a three-image panorama.
## Key Features
- Feature Detection & Description: Utilizes ORB (Oriented FAST and Rotated BRIEF) and SIFT (Scale-Invariant Feature Transform) to identify and describe keypoints.
- Descriptor Matching: Implements both Brute-Force (BF) matching and FLANN-based (Fast Library for Approximate Nearest Neighbors) matching.
- Dynamic Matching Visualization: Displays the "Best $N$ Matches" between two images, adjustable via an interactive trackbar.
- Image Stitching: Automatically stitches three consecutive images into a single panoramic view using homography estimation.
## Objective
The primary goal of this assignment is to master the pipeline for geometric computer vision tasks. Key technical objectives include:
- Comparing the efficiency and accuracy of ORB vs. SIFT descriptors.
- Implementing a Ratio Test to filter out ambiguous matches and improve robustness.
- Understanding the Homography pipeline: finding keypoints, matching descriptors, computing the homography matrix, and warping images to a common plane.
## Requirements
- Python 3.x
- OpenCV (opencv-python and opencv-contrib-python)
- NumPy
- Data Folder: images/ folder containing the source images (e.g., i1.jpg, i2.jpg, etc.) must be located in the same directory as the script.
## How to Execute the Code
```bash
python ASSIGNMENT2.py
```
