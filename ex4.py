import cv2
import numpy as np
from matplotlib import pyplot as plt


def harris_corner_detector(im, k=0.04, threshold=0.01):# TODO delete
    """
    Detect Harris corners in a grayscale image using OpenCV.

    Args:
        im (np.ndarray): Input grayscale image (H x W).
        k (float): Harris detector free parameter (default=0.04).
        threshold (float): Threshold for corner detection (default=0.01).

    Returns:
        np.ndarray: Array of shape (N, 2) containing [x, y] coordinates of detected corners.
    """
    # Convert image to float32 (required by cv2.cornerHarris)
    im_float32 = np.float32(im)

    # Apply Harris corner detection
    harris_response = cv2.cornerHarris(im_float32, blockSize=2, ksize=3, k=k)

    # Normalize and threshold the response
    harris_response = cv2.dilate(harris_response, None)  # Optional: enhances corner points
    threshold_value = threshold * harris_response.max()
    corner_mask = harris_response > threshold_value

    # Get coordinates of corners
    y_coords, x_coords = np.where(corner_mask)
    corners = np.column_stack((x_coords, y_coords))  # Shape (N, 2), format [x, y]

    return corners

def find_features_SIFT(pyr):
    """
    Detect keypoints and extract descriptors using SIFT.

    Args:
        pyr (list): Gaussian pyramid (list of images, where pyr[0] is the original).

    Returns:
        list: [keypoints, descriptors]
            - keypoints: np.ndarray (N, 2) of [x, y] coordinates.
            - descriptors: np.ndarray (N, K, K) where K=7 (simulating MOPS-like).
    """
    # Use the third level of the pyramid (pyr[2])
    img = pyr[2]

    # Initialize SIFT
    sift = cv2.SIFT_create()

    # Detect keypoints and compute descriptors
    kps, descs = sift.detectAndCompute(img, None)

    # Convert keypoints to [x, y] format (N, 2)
    keypoints = np.array([kp.pt for kp in kps], dtype=np.float32)

    # Since SIFT descriptors are 128D, we reshape to (N, K, K) where K=7
    # Here, we simulate MOPS-like descriptors by taking the first 49 values (7x7)
    K = 7
    descriptors = descs[:, :K * K].reshape(-1, K, K)

    return [keypoints, descriptors]


def match_features_flann(desc1, desc2, min_score=0.5):
    """
    Match descriptors using FLANN (faster for large datasets).
    """
    desc1_flat = desc1.reshape(len(desc1), -1).astype(np.float32)
    desc2_flat = desc2.reshape(len(desc2), -1).astype(np.float32)

    # FLANN parameters (optimized for SIFT/ORB)
    flann = cv2.FlannBasedMatcher(
        dict(algorithm=1, trees=5),  # KD-Tree for SIFT
        dict(checks=50)  # Higher = more accurate but slower
    )

    # Match descriptors (k=2 for ratio test)
    matches = flann.knnMatch(desc1_flat, desc2_flat, k=2)

    # Lowe's ratio test to filter weak matches
    good_matches = []
    for m, n in matches:
        if m.distance < min_score * n.distance:
            good_matches.append(m)

    # Extract indices
    match_ind1 = np.array([m.queryIdx for m in good_matches])
    match_ind2 = np.array([m.trainIdx for m in good_matches])

    return match_ind1, match_ind2


def get_frames(video_path, from_frame, to_frame):
    """
    Extract frames from a video file.

    Args:
        video_path (str): Path to the video file.
        from_frame (int): Starting frame index.
        to_frame (int): Ending frame index.

    Returns:
        array: List of frames as numpy arrays.
    """
    cap = cv2.VideoCapture(video_path)
    frames = []

    # Set the starting frame
    cap.set(cv2.CAP_PROP_POS_FRAMES, from_frame)

    # Read frames until the end or until to_frame is reached
    for i in range(from_frame, to_frame):
        ret, frame = cap.read()
        if not ret:
            break
        frames.append(frame)

    cap.release()
    return frames


def apply_homography(pos1, H):
    """
    Apply homography H to points pos1.

    Args:
        pos1 (np.ndarray): Points in Image1, shape (N, 2).
        H (np.ndarray): Homography matrix (3x3).

    Returns:
        np.ndarray: Transformed points in Image2, shape (N, 2).
    """
    # Convert to homogeneous coordinates
    ones = np.ones((pos1.shape[0], 1))
    pos1_hom = np.hstack((pos1, ones))  # Shape (N, 3)

    # Apply homography
    pos2_hom = H @ pos1_hom.T  # Shape (3, N)
    pos2_hom /= pos2_hom[2, :]  # Normalize by z

    return pos2_hom[:2, :].T  # Return (N, 2)


def ransac_homography(pos1, pos2, num_iters=5000, inlier_tol=2.0, translation_only=False):
    """
    Estimate homography using RANSAC.

    Args:
        pos1 (np.ndarray): Points in Image1, shape (N, 2).
        pos2 (np.ndarray): Corresponding points in Image2, shape (N, 2).
        num_iters (int): Number of RANSAC iterations.
        inlier_tol (float): Max pixel error to consider a match an inlier.
        translation_only (bool): If True, estimate only translation (no rotation).

    Returns:
        tuple: (H, inliers)
            - H: Best homography matrix (3x3).
            - inliers: Indices of inlier matches.
    """
    # Convert to OpenCV-friendly format
    pos1_cv = pos1.reshape(-1, 1, 2).astype(np.float32)
    pos2_cv = pos2.reshape(-1, 1, 2).astype(np.float32)

    # Estimate homography with RANSAC
    method = cv2.SCORE_METHOD_RANSAC
    if translation_only:
        method = cv2.RHO  # Slower but better for translation-only
    H, inliers = cv2.findHomography(
        pos1_cv, pos2_cv,
        method=method,
        ransacReprojThreshold=inlier_tol,
        maxIters=num_iters
    )

    # Convert inliers to indices
    inliers = np.where(inliers.flatten() == 1)[0]
    return H, inliers


def compute_bounding_box(homography, w, h):
    """
    Calculate the bounding box after applying homography to an image of size (w, h).

    Args:
        homography (np.ndarray): 3x3 homography matrix.
        w (int): Width of the input image.
        h (int): Height of the input image.

    Returns:
        np.ndarray: [[x_min, y_min], [x_max, y_max]].
    """
    # Corners of the original image
    corners = np.array([
        [0, 0],  # Top-left
        [w - 1, 0],  # Top-right
        [w - 1, h - 1],  # Bottom-right
        [0, h - 1]  # Bottom-left
    ])

    # Transform corners using homography
    warped_corners = apply_homography(corners, homography)

    # Get bounding box
    x_min, y_min = np.floor(warped_corners.min(axis=0)).astype(int)
    x_max, y_max = np.ceil(warped_corners.max(axis=0)).astype(int)

    return np.array([[x_min, y_min], [x_max, y_max]])


def warp_image(image, homography):
    """
    Warp an image using a homography matrix.

    Args:
        image (np.ndarray): Input image (H, W, C).
        homography (np.ndarray): 3x3 homography matrix.

    Returns:
        np.ndarray: Warped image.
    """
    h, w = image.shape[:2]

    # Compute bounding box
    [[x_min, y_min], [x_max, y_max]] = compute_bounding_box(homography, w, h)

    # Adjust homography to avoid negative coordinates
    translation = np.array([[1, 0, -x_min],
                            [0, 1, -y_min],
                            [0, 0, 1]])
    adjusted_homography = translation @ homography

    # Warp the image
    warped = cv2.warpPerspective(
        image, adjusted_homography,
        (x_max - x_min, y_max - y_min),
        flags=cv2.INTER_CUBIC
    )

    return warped


def stitch_images(image1, image2, homography):
    """
    Stitch two images using a homography matrix.

    Args:
        image1 (np.ndarray): First image (H, W, C).
        image2 (np.ndarray): Second image (H, W, C).
        homography (np.ndarray): 3x3 homography matrix from image1 to image2.

    Returns:
        np.ndarray: Stitched panorama.
    """
    # Warp image1
    warped = warp_image(image1, homography)

    # Compute offset for image2
    [[x_min, y_min], [x_max, y_max]] = compute_bounding_box(homography, image1.shape[1], image1.shape[0])
    offset_x, offset_y = -x_min, -y_min

    # Create panorama canvas
    panorama_h = max(warped.shape[0], image2.shape[0] + offset_y)
    panorama_w = max(warped.shape[1], image2.shape[1] + offset_x)
    panorama = np.zeros((panorama_h, panorama_w, 3), dtype=np.uint8)

    # Place warped image1
    panorama[:warped.shape[0], :warped.shape[1]] = warped

    # Place image2 (with blending in overlap region)
    mask = (image2 > 0).all(axis=2)  # Ignore black pixels
    panorama[offset_y:offset_y + image2.shape[0],
    offset_x:offset_x + image2.shape[1]][mask] = image2[mask]

    return panorama


def accumulate_homographies(H_successive, m):
    """
    Convert pairwise homographies to global homographies w.r.t. the reference frame `m`.

    Args:
        H_successive (list): List of M-1 homographies, where H_successive[i] maps image i to i+1.
        m (int): Index of the reference image (usually middle image).

    Returns:
        list: M homographies, where H2m[i] maps image i to the reference frame m.
    """
    M = len(H_successive) + 1
    H2m = [np.eye(3)] * M  # Initialize with identity matrices

    # Forward accumulation (for images before m: i < m)
    for i in range(m - 1, -1, -1):
        H2m[i] = H2m[i + 1] @ H_successive[i]

    # Backward accumulation (for images after m: i > m)
    for i in range(m + 1, M):
        H2m[i] = H2m[i - 1] @ np.linalg.inv(H_successive[i - 1])

    # Normalize homographies (ensure H[2,2] == 1)
    for i in range(M):
        H2m[i] /= H2m[i][2, 2]

    return H2m


def compute_panorama_bounds(images, H2m):
    """
    Compute the bounding box for the final panorama.

    Args:
        images (list): List of input images.
        H2m (list): List of homographies mapping each image to the reference frame.

    Returns:
        tuple: (x_min, y_min, x_max, y_max).
    """
    all_corners = []
    for i, (img, H) in enumerate(zip(images, H2m)):
        h, w = img.shape[:2]
        corners = np.array([[0, 0], [w - 1, 0], [w - 1, h - 1], [0, h - 1]])
        warped_corners = apply_homography(corners, H)
        all_corners.append(warped_corners)

    all_corners = np.vstack(all_corners)
    x_min, y_min = np.floor(all_corners.min(axis=0)).astype(int)
    x_max, y_max = np.ceil(all_corners.max(axis=0)).astype(int)
    return x_min, y_min, x_max, y_max


def warp_all_images(images, H2m):
    """
    Warp all images to the reference frame and blend into a panorama.

    Args:
        images (list): List of input images.
        H2m (list): List of homographies mapping each image to the reference frame.

    Returns:
        np.ndarray: Stitched panorama.
    """
    # Compute panorama bounds
    x_min, y_min, x_max, y_max = compute_panorama_bounds(images, H2m)
    panorama_w = x_max - x_min
    panorama_h = y_max - y_min

    # Initialize panorama canvas
    panorama = np.zeros((panorama_h, panorama_w, 3), dtype=np.float32)
    panorama_mask = np.zeros((panorama_h, panorama_w), dtype=np.float32)

    # Warp and blend each image
    for i, (img, H) in enumerate(zip(images, H2m)):
        # Adjust homography for panorama offset
        T = np.array([[1, 0, -x_min],
                      [0, 1, -y_min],
                      [0, 0, 1]])
        H_adjusted = T @ H

        # Warp image
        warped = cv2.warpPerspective(
            img, H_adjusted, (panorama_w, panorama_h),
            flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_TRANSPARENT
        )

        # Create mask (non-black regions)
        mask = (warped.sum(axis=2) > 0).astype(np.float32)

        # Blend into panorama (weighted average)
        panorama += warped
        panorama_mask += mask

    # Normalize by mask to avoid division by zero
    panorama_mask[panorama_mask == 0] = 1  # Avoid division by zero
    panorama /= panorama_mask[..., np.newaxis]

    return panorama.astype(np.uint8)


def build_gaussian_pyramid(image, max_levels=3):

    pyramid = [image]
    for i in range(max_levels - 1):
        image = cv2.pyrDown(image)
        pyramid.append(image)

    return pyramid


def precompute_warps_and_boxes(images, homographies):
    """
    Warp all images and compute their bounding boxes in the panorama coordinate system.

    Args:
        images (list): List of input images.
        homographies (list): List of homographies (H_i→ref).

    Returns:
        tuple: (warped_images, bounding_boxes)
    """
    warped_images = []
    bounding_boxes = np.zeros((len(images), 2, 2))

    for i, (img, H) in enumerate(zip(images, homographies)):
        warped = warp_image(img, H)
        warped_images.append(warped)
        bounding_boxes[i] = compute_bounding_box(H, img.shape[1], img.shape[0])

    return warped_images, bounding_boxes


def compute_strip_boundaries(warped_slice_centers, panorama_width):
    """
    Calculate x-coordinates where strips start/end in the panorama.

    Args:
        warped_slice_centers (np.ndarray): Shape (N_panoramas, N_images).
        panorama_width (int): Width of the panorama canvas.

    Returns:
        np.ndarray: Boundaries array (N_panoramas, N_images + 1).
    """
    boundaries = np.zeros((warped_slice_centers.shape[0], warped_slice_centers.shape[1] + 1))
    boundaries[:, 1:-1] = (warped_slice_centers[:, :-1] + warped_slice_centers[:, 1:]) / 2
    boundaries[:, -1] = panorama_width
    return boundaries.round().astype(np.int64)


def build_panoramas(warped_images, bounding_boxes, strip_boundaries):
    """
    Combine strips from warped images into panoramas.

    Args:
        warped_images (list): List of warped images.
        bounding_boxes (np.ndarray): Bounding boxes of warped images.
        strip_boundaries (np.ndarray): Strip boundaries (N_panoramas, N_images + 1).

    Returns:
        np.ndarray: Panoramas array (N_panoramas, H, W, 3).
    """
    panorama_size = np.max(bounding_boxes, axis=(0, 1)).astype(np.int64) + 1
    panoramas = np.zeros((strip_boundaries.shape[0], panorama_size[1], panorama_size[0], 3), dtype=np.uint8)

    for panorama_idx in range(strip_boundaries.shape[0]):
        for img_idx in range(len(warped_images)):
            x_start, x_end = strip_boundaries[panorama_idx, img_idx], strip_boundaries[panorama_idx, img_idx + 1]
            x_offset, y_offset = bounding_boxes[img_idx, 0].astype(np.int64)

            # Skip if strip has zero width
            if x_start >= x_end:
                continue

            # Calculate strip coordinates in warped image
            strip_x_start = x_start - x_offset
            strip_x_end = x_end - x_offset

            # Ensure coordinates are within warped image bounds
            strip_x_start = max(strip_x_start, 0)
            strip_x_end = min(strip_x_end, warped_images[img_idx].shape[1])

            # Skip if strip would be empty
            if strip_x_start >= strip_x_end:
                continue

            # Extract strip from warped image
            strip = warped_images[img_idx][:, strip_x_start:strip_x_end]

            # Calculate target y-coordinates
            y_start = max(y_offset, 0)
            y_end = min(y_offset + strip.shape[0], panorama_size[1])
            strip_y_start = max(-y_offset, 0)
            strip_y_end = strip_y_start + (y_end - y_start)

            # Resize only if necessary and possible
            target_width = x_end - x_start
            if strip.shape[1] != target_width and strip.size > 0:
                try:
                    strip = cv2.resize(strip, (target_width, strip.shape[0]))
                except:
                    continue

            print(f"Image {img_idx}:")
            print(f"  Warped image shape: {warped_images[img_idx].shape}")
            print(f"  Bounding box: {bounding_boxes[img_idx]}")
            print(f"  Strip boundaries: {x_start} to {x_end}")
            print(f"  Calculated strip coords: {strip_x_start} to {strip_x_end}")
            print(f"  Strip shape: {strip.shape if 'strip' in locals() else 'N/A'}")
            print(f"  Panorama coords: {y_start} to {y_end}")
            print(f"  Strip y coords: {strip_y_start} to {strip_y_end}")
            print(f"  Panorama shape: {panoramas[panorama_idx].shape}")
            print(f"  Strip y shape: {strip[strip_y_start:strip_y_end].shape}")

            # Paste if dimensions match and strip is valid
            if (y_start < y_end and strip_y_start < strip_y_end and
                strip.shape[1] == target_width and strip.size > 0):
                panoramas[panorama_idx, y_start:y_end, x_start:x_end] = strip[strip_y_start:strip_y_end]

    return panoramas

def stitch_multiple_images(images):
    """
    Stitch multiple images using the strip-based approach.

    Args:
        images (list): List of input images (ordered left-to-right).

    Returns:
        np.ndarray: Stitched panorama.
    """
    # Step 1: Compute pairwise homographies (H_i→i+1)
    H_successive = []
    for i in range(len(images) - 1):
        kp1, desc1 = find_features_SIFT(build_gaussian_pyramid(images[i]))
        kp2, desc2 = find_features_SIFT(build_gaussian_pyramid(images[i + 1]))
        matches1, matches2 = match_features_flann(desc1, desc2)
        H, _ = ransac_homography(kp1[matches1], kp2[matches2])
        H_successive.append(H)

    # Step 2: Accumulate homographies to reference frame (middle image)
    ref_idx = len(images) // 2
    H2ref = accumulate_homographies(H_successive, ref_idx)

    # Step 3: Warp all images and compute bounding boxes
    warped_images, bounding_boxes = precompute_warps_and_boxes(images, H2ref)

    # Step 4: Compute strip boundaries (for 1 panorama)
    slice_centers = np.array([img.shape[1] // 2 for img in images])  # Middle column of each image
    warped_centers = np.array([apply_homography(np.array([[c, 0]]), H) for c, H in zip(slice_centers, H2ref)])
    warped_slice_centers = warped_centers[:, 0, 0].reshape(1, -1)  # Shape (1, N_images)

    panorama_width = int(np.max(bounding_boxes[:, :, 0])) + 1
    strip_boundaries = compute_strip_boundaries(warped_slice_centers, panorama_width)

    # Step 5: Build panorama
    panoramas = build_panoramas(warped_images, bounding_boxes, strip_boundaries)[0]  # Get first panorama
    return panoramas

# Test functions

def harris_corner_test():# TODO delete
    # Load a grayscale image
    image = cv2.imread("example.jpg", cv2.IMREAD_GRAYSCALE)

    # Detect corners
    corners = harris_corner_detector(image)

    # Display results
    plt.imshow(image, cmap='gray')
    plt.scatter(corners[:, 0], corners[:, 1], s=5, c='red', marker='o')
    plt.title("Harris Corners Detected")
    plt.show()
    # Save the image with corners
    cv2.imwrite("output_corners_detected.jpg", image)



def feature_detect_test():
    image = cv2.imread(r"C:\Users\amir\Documents\University Files\IMPR\ex4\First_shower.jpg", cv2.IMREAD_GRAYSCALE)
    # Build a Gaussian pyramid with 3 levels
    pyramid = [image]
    for i in range(2):
        image = cv2.pyrDown(image)
        pyramid.append(image)
    # Detect features
    keypoints, descriptors = find_features_SIFT(pyramid)
    print(f"Detected {len(keypoints)} keypoints")
    print(f"Descriptor shape: {descriptors.shape}")  # Expected: (N, 7, 7)

    # Draw keypoints
    image = cv2.cvtColor(pyramid[2], cv2.COLOR_GRAY2BGR)
    for kp in keypoints:
        x, y = kp
        cv2.circle(image, (int(x), int(y)), 1, (0, 255, 0), -1)

    # plot Features detected
    plt.imshow(image)
    plt.title("Features Detected")
    plt.show()
    # Save the image with features
    cv2.imwrite("output_features_detected.jpg", image)


    # Draw matches (for debugging)

    img1 = cv2.imread(r"C:\Users\amir\Documents\University Files\IMPR\ex4\Nivi_1.jpg", cv2.IMREAD_GRAYSCALE)
    img2 = cv2.imread(r"C:\Users\amir\Documents\University Files\IMPR\ex4\Nivi_2.jpg", cv2.IMREAD_GRAYSCALE)



    # Build a Gaussian pyramid with 3 levels
    pyramid1 = [img1]
    for i in range(2):
        img1 = cv2.pyrDown(img1)
        pyramid1.append(img1)
    pyramid2 = [img2]
    for i in range(2):
        img2 = cv2.pyrDown(img2)
        pyramid2.append(img2)



    # Detect keypoints and descriptors (using SIFT/ORB)
    kp1, desc1 = find_features_SIFT(pyramid1)
    kp2, desc2 = find_features_SIFT(pyramid2)

    print(f"Detected {len(kp1)} keypoints in image 1")
    print(f"Detected {len(kp2)} keypoints in image 2")

    # Match features
    match_ind1, match_ind2 = match_features_flann(desc1, desc2, min_score=0.5)

    # Convert keypoints to OpenCV format
    cv_kp1 = [cv2.KeyPoint(x, y, 1) for x, y in kp1]
    cv_kp2 = [cv2.KeyPoint(x, y, 1) for x, y in kp2]

    # Draw matches with lines
    matches = [cv2.DMatch(i, j, 0) for i, j in zip(match_ind1, match_ind2)]
    print(f"Number of DMatches created: {len(matches)}")

    # Try drawing with default flags
    match_image = cv2.drawMatches(pyramid1[2], cv_kp1, pyramid2[2], cv_kp2, matches, None, (0, 255, 0), (0, 0, 255), flags=cv2.DRAW_MATCHES_FLAGS_DEFAULT)

    plt.figure(figsize=(12, 6))
    plt.imshow(match_image)
    plt.title("Matches")
    plt.show()
    # Save the image with matches
    cv2.imwrite("output_matches.jpg", match_image)

    # Display matches with lines
    # Convert keypoints to numpy arrays
    pos1 = np.array(kp1)[match_ind1]
    pos2 = np.array(kp2)[match_ind2]
    H, inliers = ransac_homography(pos1, pos2)
    # Display matches
    display_matches(pyramid1[2], pyramid2[2], kp1, kp2, inliers)


def display_matches(im1, im2, pos1, pos2, inliers):
    """
    Display matches (yellow = inliers, blue = outliers).
    """
    # Stack images horizontally
    im_combined = np.hstack((im1, im2))

    plt.imshow(im_combined, cmap='gray')

    # Plot all matches (blue)
    for i in range(len(pos1)):
        x1, y1 = pos1[i]
        x2, y2 = pos2[i] + np.array([im1.shape[1], 0])  # Offset Image2
        plt.plot([x1, x2], [y1, y2], 'b-', linewidth=1.0, alpha=0.3)

    # Highlight inliers (yellow)
    for i in inliers:
        x1, y1 = pos1[i]
        x2, y2 = pos2[i] + np.array([im1.shape[1], 0])
        plt.plot([x1, x2], [y1, y2], 'y-', linewidth=1.5)

    plt.show()
    # Save the image with matches
    cv2.imwrite("output_matches_with_lines.jpg", im_combined)

def stitch_images_test(image1, image2):

    pyramid1 = [image1]
    for i in range(2):
        image1 = cv2.pyrDown(image1)
        pyramid1.append(image1)
    pyramid2 = [image2]
    for i in range(2):
        image2 = cv2.pyrDown(image2)
        pyramid2.append(image2)

        #detect keypoints and descriptors (using SIFT/ORB)
    kp1, desc1 = find_features_SIFT(pyramid1)
    kp2, desc2 = find_features_SIFT(pyramid2)

    # Match features
    match_ind1, match_ind2 = match_features_flann(desc1, desc2, min_score=0.5)
    # Convert keypoints to OpenCV format
    cv_kp1 = [cv2.KeyPoint(x, y, 1) for x, y in kp1]
    cv_kp2 = [cv2.KeyPoint(x, y, 1) for x, y in kp2]

    matches = [cv2.DMatch(i, j, 0) for i, j in zip(match_ind1, match_ind2)]

    pos1 = np.array(kp1)[match_ind1]
    pos2 = np.array(kp2)[match_ind2]


    H, inliers = ransac_homography(pos1, pos2)

    # Assume we already computed homography H (from RANSAC)
    H = np.array(H)

    # Stitch images
    panorama = stitch_images(image1, image2, H)

    # apply the homography on image 1
    warped_image1 = warp_image(image1, H)
    # resize the warped image to match image 2
    warped_image1 = cv2.resize(warped_image1, (image2.shape[1], image2.shape[0]))
    # subtract the warped image from image 2
    diff = cv2.absdiff(image2, warped_image1)
    # display the difference
    plt.imshow(diff)
    plt.title("Difference between image2 and warped image1")
    plt.axis('off')
    plt.show()
    # Save the image with difference
    cv2.imwrite("output_difference.jpg", diff)

    # subtract image 1 from image 2
    diff2 = cv2.absdiff(image2, image1)
    # display the difference
    plt.imshow(diff2)
    plt.title("Difference between image2 and image1")
    plt.axis('off')
    plt.show()
    # Save the image with difference
    cv2.imwrite("output_difference2.jpg", diff2)



    # Plot the stitched image
    plt.imshow(panorama)
    plt.title("Stitched Image")
    plt.axis('off')
    plt.show()
    # Save the stitched image
    cv2.imwrite("output_stitched_image.jpg", panorama)
    return panorama

def stitch_multiple_images_test(images):
    # Assume images is a list of images to stitch
    if len(images) < 2:
        raise ValueError("At least two images are required for stitching.")

    panorama = stitch_multiple_images(images)

    return panorama

def get_frames_from_video(video_path, start_frame=0, end_frame=3):
    # returns an numpy array of frames from the video
    cap = cv2.VideoCapture(video_path)
    frames = []
    for i in range(start_frame, end_frame):
        cap.set(cv2.CAP_PROP_POS_FRAMES, i)
        ret, frame = cap.read()
        if not ret:
            break
        frames.append(frame)
    cap.release()
    return frames


def tests_everything_so_far():
    video_path = r"C:\Users\amir\Documents\University Files\IMPR\ex4\Exercise Inputs-20250201\boat.mp4"
    frames = get_frames(video_path, 0, 3)
    feature_detect_test()

    # plot frames
    plt.imshow(frames[1])
    plt.title("Frame 0")
    plt.axis('off')
    plt.show()
    # Save the image with frame
    cv2.imwrite("output_frame_0.jpg", frames[0])

    plt.imshow(frames[2])
    plt.title("Frame 1")
    plt.axis('off')
    plt.show()
    # Save the image with frame
    cv2.imwrite("output_frame_1.jpg", frames[1])

    # Test stitching images
    stitch_images_test(frames[0], frames[1])

    # Test stitching multiple images
    frames = get_frames(video_path, 0, 3)
    panorama = stitch_multiple_images_test(frames)
    # Display the stitched panorama
    plt.imshow(panorama)
    plt.title("Stitched Panorama- 3 Frames")
    plt.axis('off')
    plt.show()
    # Save the stitched image
    cv2.imwrite("output_stitched_image_3_frames.jpg", panorama)

    frames_0_to_100 = get_frames_from_video(video_path, 0, 100)

    panorama = stitch_multiple_images(frames_0_to_100)
    # plot the panorama
    plt.imshow(panorama)
    plt.title("Stitched Panorama - 100 Frames")
    plt.axis('off')
    plt.show()
    # Save the stitched image
    cv2.imwrite("output_stitched_image_100_frames.jpg", panorama)

    # Load 3 test frames
    frames = get_frames(video_path, 0, 3)  # First 3 frames
    image1, image2, image3 = [cv2.cvtColor(f, cv2.COLOR_BGR2GRAY) for f in frames]

    # Test feature detection and matching between frame1 and frame2
    pyr1 = build_gaussian_pyramid(image1)
    pyr2 = build_gaussian_pyramid(image2)

    # Detect keypoints and descriptors (using SIFT/ORB)
    kp1, desc1 = find_features_SIFT(pyr1)
    kp2, desc2 = find_features_SIFT(pyr2)

    print(f"Detected {len(kp1)} keypoints in image 1")
    print(f"Detected {len(kp2)} keypoints in image 2")

    # Match features
    match_ind1, match_ind2 = match_features_flann(desc1, desc2, min_score=0.5)

    # Convert keypoints to OpenCV format
    cv_kp1 = [cv2.KeyPoint(x, y, 1) for x, y in kp1]
    cv_kp2 = [cv2.KeyPoint(x, y, 1) for x, y in kp2]

    # Draw matches with lines
    matches = [cv2.DMatch(i, j, 0) for i, j in zip(match_ind1, match_ind2)]
    print(f"Number of DMatches created: {len(matches)}")

    # Try drawing with default flags
    match_image = cv2.drawMatches(pyr1[2], cv_kp1, pyr2[2], cv_kp2, matches, None)

    plt.figure(figsize=(12, 6))
    plt.imshow(match_image)
    plt.title("Matches")
    plt.show()
    # Save the image with matches
    cv2.imwrite("output_matches.jpg", match_image)

    # Display matches with lines
    # Convert keypoints to numpy arrays
    pos1 = np.array(kp1)[match_ind1]
    pos2 = np.array(kp2)[match_ind2]
    H, inliers = ransac_homography(pos1, pos2)


    inliers = np.array([i for i in range(len(matches)) if i in inliers])

    # Display matches
    display_matches(pyr1[2], pyr2[2], pos1, pos2, inliers)

    # print the maximum distance between the matches
    max_distance = np.max([m.distance for m in matches])
    print(f"Max distance between matches: {max_distance}")
    # print the maximum distance between the inliers
    max_distance_inliers = np.max([m.distance for m in matches if m.queryIdx in inliers])
    print(f"Max distance between inliers: {max_distance_inliers}")


    H, inliers = ransac_homography(pos1, pos2)
    print("Homography Matrix:\n", np.round(H, 2))

    # Visualize inliers (yellow) vs. outliers (blue)
    plt.figure(figsize=(15, 5))
    combined = np.hstack([image1, image2])
    plt.imshow(combined, cmap='gray')
    plt.scatter(pos1[:, 0], pos1[:, 1], c='b', s=5, label='Outliers')
    plt.scatter(pos1[inliers, 0], pos1[inliers, 1], c='y', s=5, label='Inliers')
    plt.scatter(pos2[:, 0] + image1.shape[1], pos2[:, 1], c='b', s=5)
    plt.scatter(pos2[inliers, 0] + image1.shape[1], pos2[inliers, 1], c='y', s=5)
    plt.title("Inliers (Yellow) vs. Outliers (Blue)")
    plt.legend()
    plt.show()
    # Save the image with inliers and outliers
    cv2.imwrite("output_inliers_outliers.jpg", combined)

    # Compute pairwise homographies
    H12, _ = ransac_homography(kp1[match_ind1], kp2[match_ind2])
    kp3, desc3 = find_features_SIFT(build_gaussian_pyramid(image3))
    match_ind2_3, match_ind3 = match_features_flann(desc2, desc3, min_score=0.5)
    H23, _ = ransac_homography(kp2[match_ind2_3], kp3[match_ind3])

    # Accumulate to middle frame (frame2)
    H_successive = [H12, H23]
    H2m = accumulate_homographies(H_successive, m=1)  # m=1 is middle frame (0-based index)

    print("H1→2 (Frame1 to Frame2):\n", np.round(H_successive[0], 2))
    print("H2→m (Frame2 to Middle=Frame2):\n", np.round(H2m[1], 2))  # Should be identity
    print("H3→m (Frame3 to Middle=Frame2):\n", np.round(H2m[2], 2))  # Should ≈ inverse(H23)

    # Compute bounding boxes for all frames
    w, h = image1.shape[1], image1.shape[0]
    bbox1 = compute_bounding_box(H2m[0], w, h)  # Frame1 → Middle
    bbox2 = compute_bounding_box(H2m[1], w, h)  # Frame2 → Middle (should be [0,0]-[w,h])
    bbox3 = compute_bounding_box(H2m[2], w, h)  # Frame3 → Middle

    print("Frame1 BBox:", bbox1)
    print("Frame2 BBox:", bbox2)  # Should be [[0, 0], [w-1, h-1]]
    print("Frame3 BBox:", bbox3)

    # Warp Frame1 and Frame3 to middle frame's space
    warped1 = warp_image(frames[0], H2m[0])
    warped3 = warp_image(frames[2], H2m[2])

    # Visualize alignment
    plt.figure(figsize=(15, 5))
    plt.subplot(131);
    plt.imshow(warped1);
    plt.title("Frame1 → Middle")
    plt.subplot(132);
    plt.imshow(frames[1]);
    plt.title("Middle Frame (Frame2)")
    plt.subplot(133);
    plt.imshow(warped3);
    plt.title("Frame3 → Middle")
    plt.show()
    # Save the images with warped frames
    cv2.imwrite("output_warped_frame1.jpg", warped1)

    # create a panorama from Trees video
    tree_video_path = r"C:\Users\amir\Documents\University Files\IMPR\ex4\Exercise Inputs-20250201\Trees.mp4"
    frames = get_frames(tree_video_path, 0, 100)
    # rotate the frames 90 degrees to the right
    frames = [cv2.rotate(frame, cv2.ROTATE_90_CLOCKWISE) for frame in frames]
    # Stitch the frames together
    panorama = stitch_multiple_images(frames)
    # rotate the panorama 90 degrees to the left
    panorama = cv2.rotate(panorama, cv2.ROTATE_90_COUNTERCLOCKWISE)
    # Display the stitched panorama
    plt.imshow(panorama)
    plt.title("Stitched Panorama - Trees Video")
    plt.axis('off')
    plt.show()
    # Save the panorama
    cv2.imwrite("output_panorama_trees.jpg", panorama)






tests_everything_so_far()