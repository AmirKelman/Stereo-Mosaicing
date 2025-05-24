Stereo Mosaicing with SIFT and Homography Estimation
This repository contains the implementation of a stereo mosaicing algorithm for creating panoramic videos from a sequence of frames, as part of an image processing exercise. The project processes video frames (e.g., from boat.mp4 and trees.mp4) with significant overlap and left-to-right camera motion, stitching them into panoramas using SIFT for feature detection, FLANN for matching, RANSAC for homography estimation, and a strip-based approach for stitching. The accompanying report details the methodology, challenges, and results.
Overview
The goal of this project is to stitch video frames into panoramic videos, handling both dynamic scenes (e.g., moving trees) and changing viewpoints (e.g., a boat moving across water). Key components include:

Feature Detection and Description: Using SIFT to detect keypoints and compute descriptors on Gaussian pyramid level 2.
Feature Matching: Matching keypoints across frames with FLANN and Lowe's ratio test.
Homography Estimation: Estimating homographies between consecutive frames using RANSAC.
Image Warping and Stitching: Warping frames to a common coordinate system (middle frame) and stitching them into panoramas using a strip-based approach or weighted averaging.

Report Findings

Feature detection and matching were robust, as shown in visualizations (e.g., Figures 1 and 2 in the report).
Challenges arose in homography estimation, leading to misalignments (Figure 4).
Stitching errors caused panorama drift and visible seams, especially in 100-frame sequences (Figures 6 and 7).

Files
Code

ex4.py: Main implementation of the stereo mosaicing pipeline, including feature detection (find_features_SIFT), matching (match_features_flann), homography estimation (ransac_homography), and stitching (stitch_multiple_images).

Report

Image Processing - Exercise 4 - Amir Kelman.pdf: Detailed report covering the algorithm, implementation, visualizations, challenges, and results.

Output Files
The code generates several output images during testing (saved in the working directory):

output_corners_detected.jpg: Harris corner detection result (test function, marked for deletion).
output_features_detected.jpg: Keypoints detected using SIFT on First_shower.jpg.
output_matches.jpg: Feature matches between Nivi_1.jpg and Nivi_2.jpg.
output_matches_with_lines.jpg: Matches with inlier/outlier lines.
output_difference.jpg and output_difference2.jpg: Difference images showing alignment errors.
output_stitched_image.jpg: Stitched panorama from two frames.
output_stitched_image_3_frames.jpg: Panorama from 3 frames of boat.mp4.
output_stitched_image_100_frames.jpg: Panorama from 100 frames of boat.mp4.
output_inliers_outliers.jpg: Inliers vs. outliers visualization.
output_warped_frame1.jpg: Warped frame from boat.mp4.
output_panorama_trees.jpg: Panorama from 100 frames of trees.mp4.

Resources

Input files (not included due to size, referenced in the code):
boat.mp4 and trees.mp4: Video inputs from the course materials.
First_shower.jpg, Nivi_1.jpg, Nivi_2.jpg: Test images used for feature detection and matching.



Setup
Prerequisites

Python 3.x
Libraries:
opencv-python (for SIFT, FLANN, RANSAC, and image processing)
numpy (for array operations)
matplotlib (for visualization)



Installation

Clone the repository:git clone https://github.com/AmirKelman/Stereo-Mosaicing.git
cd Stereo-Mosaicing


Install dependencies in a virtual environment (recommended):python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
pip install opencv-python numpy matplotlib


Place input files (boat.mp4, trees.mp4, First_shower.jpg, Nivi_1.jpg, Nivi_2.jpg) in the appropriate directory (e.g., C:\Users\amir\Documents\University Files\IMPR\ex4\) or update the file paths in ex4.py.

Usage
Running the Code

The main script ex4.py includes test functions that demonstrate the entire pipeline. Run the script to execute all tests:python ex4.py


Key test function: tests_everything_so_far() performs the following:
Extracts frames from boat.mp4 and trees.mp4.
Tests feature detection on First_shower.jpg.
Matches features between Nivi_1.jpg and Nivi_2.jpg.
Stitches frames into panoramas (3 frames and 100 frames from boat.mp4, 100 frames from trees.mp4).
Visualizes results (e.g., keypoints, matches, inliers/outliers, panoramas).



Customization

Modify file paths in tests_everything_so_far() to use different input videos or images.
Adjust hyperparameters in the code:
find_features_SIFT: Uses pyramid level 2 (can be changed by modifying pyr[2]).
match_features_flann: min_score=0.5 (Lowe's ratio test threshold).
ransac_homography: num_iters=5000, inlier_tol=2.0.



Results

Feature Detection: SIFT effectively detected keypoints, as shown in output_features_detected.jpg (Figure 1 in the report).
Feature Matching: FLANN with Lowe's ratio test identified reliable matches, visualized in output_matches.jpg (Figure 2).
Homography Estimation: RANSAC produced homographies, but errors caused misalignments (see output_inliers_outliers.jpg, Figure 3).
Stitching: Panoramas showed visible seams and drift, especially in 100-frame sequences (output_stitched_image_100_frames.jpg, Figures 6 and 7).

Challenges (from the Report)

Homography Errors: Inaccurate homographies led to misalignments (Figure 4).
Panorama Drift: Accumulated errors over many frames caused distortions.
Strip-Based Stitching: Incorrect strip boundaries resulted in incomplete panoramas.

Contributing

Feel free to fork this repository, suggest improvements, or report issues. Contributions to improve homography estimation or stitching quality are welcome.


Acknowledgments

Based on HUJI CS Image Processing course materials (Exercise 4, 2025).
Uses OpenCV for SIFT, FLANN, and RANSAC implementations.


Notes

File Paths: Update paths in ex4.py to match your local setup.
Large Files: Video inputs (boat.mp4, trees.mp4) are not included due to size constraints. Replace with your own.
Improvements: As noted in the report, future work could focus on multi-frame homography optimization, advanced blending, and adaptive thresholding.

