# Stereo Mosaicing

This project implements a stereo mosaicing pipeline that generates panoramic images from video frames captured with lateral camera motion. It was developed as part of an image processing university course.

## 📽️ Project Goal

To construct wide-view panoramas from stereo video sequences by:
- Detecting key features
- Matching them across frames
- Estimating geometric transformations
- Warping and blending the aligned images

The pipeline is designed to handle dynamic scenes, camera movement, and varying viewpoints.

---

## 🧠 Key Concepts and Techniques

- **SIFT Feature Detection:** Detect robust keypoints using OpenCV's SIFT algorithm.
- **Feature Matching:** Use FLANN-based matcher with Lowe’s ratio test to find point correspondences between frames.
- **Homography Estimation:** Estimate geometric transformations using RANSAC.
- **Gaussian Pyramid:** Build multi-scale representations to improve efficiency and robustness.
- **Image Warping:** Align images to a common coordinate system using estimated homographies.
- **Panorama Stitching:**
  - **Weighted averaging** and
  - **Strip-based blending** methods are used for combining warped frames into a panorama.

---

## 🛠️ Implementation

The main pipeline is in `ex4.py` and includes:
- `get_frames` – Extracts frames from video.
- `find_features_SIFT` – Detects keypoints and descriptors.
- `match_features_flann` – Matches features between frames.
- `ransac_homography` – Estimates transformations.
- `warp_image`, `stitch_images`, `warp_all_images` – Align and blend images.
- `accumulate_homographies` – Builds global transformations to a common reference frame.
- `stitch_multiple_images` – Main entry point for stitching entire sequences.

### Libraries Used
- Python
- OpenCV
- NumPy
- Matplotlib

---

## 🧪 Sample Results

The algorithm was tested on:
- **trees.mp4** – a dynamic scene with foliage
- **boat.mp4** – changing viewpoint with a boat scene

Despite robust feature detection and matching, stitching challenges remained due to accumulated errors in homography estimation.

See the [📄 report (PDF)](./Image%20Processing%20-%20Exercise%204%20-%20Amir%20Kelman.pdf) for detailed results, visualizations, and insights.

---

## 📈 Challenges & Future Work

- **Homography accuracy:** RANSAC tuning is critical to avoid misalignments.
- **Panorama drift:** Errors accumulate over long sequences.
- **Stitching robustness:** Strip boundaries and warp blending require refinement.

### Future improvements:
- Multi-frame homography refinement
- Seam-aware blending
- Adaptive matching thresholds

---

## 📁 File Structure

```bash
.
├── ex4.py                         # Main stereo mosaicing implementation
├── Image Processing - Exercise 4 - Amir Kelman.pdf   # Detailed report and analysis
└── README.md                      # This file
