# IVP_Image-registration-using-ORB-Algorithm
Image registration using ORB and MST algorithm with comparison of speed on GPU and CPU
# Image Registration with ORB and MST Algorithms

This project implements **Image Registration** using **ORB (Oriented FAST and Rotated BRIEF)** for feature detection and matching, combined with a **Minimum Spanning Tree (MST)**-based approach to refine the feature alignment. The method aligns two images by identifying key points and minimizing registration error.

---

## Features

- **Feature Detection and Matching**:
  - Uses ORB for detecting key points and describing features.
  - Matches features between two images using Hamming distance.
- **Transformation Estimation**:
  - Computes a homography matrix to align images.
- **Error Reduction**:
  - Refines alignment using MST to eliminate outlier matches.
- **Visualization**:
  - Displays matched keypoints and the aligned result.

---

## Prerequisites

Make sure you have the following installed:

- Python 3.x
- OpenCV
- NumPy
- Matplotlib

Install dependencies using:
```bash
pip install opencv-python numpy matplotlib
