import os
import cv2
import numpy as np

# Function to detect lines in an image using Canny edge detection and Hough line transform
def detect_lines(img, low_threshold=50, high_threshold=150, rho=1, theta=np.pi/180, threshold=100):
    # Convert the image to grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # Detect edges using Canny
    edges = cv2.Canny(gray, low_threshold, high_threshold)
    # Detect lines using Hough transform
    lines = cv2.HoughLines(edges, rho, theta, threshold)
    return lines

# Function to process the image and detect SIFT features, then match them between left and right halves
def process_img_sift(img, lines):
    h, w = img.shape[:2]
    w_half = w // 2
    left_img = img[:, :w_half]
    right_img = img[:, w_half:]

    # Create SIFT detector with specific parameters
    sift = cv2.SIFT_create(sigma=1, contrastThreshold=0.04, edgeThreshold=20, nOctaveLayers=20)
    # Detect and compute SIFT features in the left and right images
    kp1, des1 = sift.detectAndCompute(left_img, None)
    kp2, des2 = sift.detectAndCompute(right_img, None)

    # Parameters for FLANN based matcher
    FLANN_INDEX_KDTREE = 1
    index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
    search_params = dict(checks=50)
    flann = cv2.FlannBasedMatcher(index_params, search_params)
    # Match features between the left and right images
    matches = flann.knnMatch(des1, des2, k=2)

    pts1 = []
    pts2 = []
    # Apply ratio test to keep good matches
    for m, n in matches:
        pts2.append(kp2[m.trainIdx].pt)
        pts1.append(kp1[m.queryIdx].pt)

    # Convert points to integer format
    pts1 = np.int32(pts1)
    pts2 = np.int32(pts2)
    # Find fundamental matrix using RANSAC
    F, mask = cv2.findFundamentalMat(pts1, pts2, cv2.FM_RANSAC)

    # Filter points using the mask
    pts1 = pts1[mask.ravel() == 1]
    pts2 = pts2[mask.ravel() == 1]

    return np.float32(pts1), np.float32(pts2), left_img

# Function to triangulate 3D points from 2D correspondences in two images
def triangulate_points(P_left, P_right, points_left, points_right):
    # Triangulate points to get homogeneous coordinates
    points_4D = cv2.triangulatePoints(P_left, P_right, points_left.T, points_right.T)
    # Convert to 3D coordinates by dividing by the homogeneous coordinate
    points_3D = points_4D[:3] / points_4D[3]
    return points_3D.T

if __name__ == '__main__':
    input_path = "FinalProject/FinalProject/SBS images/"
    output_path = "FinalProject/output"
    # Create output directory if it doesn't exist
    if not os.path.exists(output_path):
        os.makedirs(output_path)
    output_xyz = "M11215075.xyz"
    output_txt = "M11215075.txt"
    output_xyz_path = os.path.join(output_path, output_xyz)
    output_txt_path = os.path.join(output_path, output_txt)

    # Intrinsic and extrinsic parameters for the left camera
    K_left = np.array([[1000.0, 0, 360.0], [0, 1000.0, 640.0], [0, 0, 1.0]])
    Rt_left = np.array([[0.88649035, -0.46274707, 0, -14.42428], 
                        [-0.070794605, -0.13562201, -0.98822814, 86.532959], 
                        [0.45729965, 0.8760547, -0.1529876, 235.35446]])
    P_left = K_left @ Rt_left

    # Intrinsic and extrinsic parameters for the right camera
    K_right = np.array([[1100.0, 0, 360.0], [0, 1100.0, 640.0], [0, 0, 1.0]]) 
    Rt_right = np.array([[0.98480779, -0.17364818, 0, -0.98420829],
                         [-0.026566068, -0.15066338, -0.98822814, 85.070221],
                         [0.17160401, 0.97321475, -0.1529876, 236.97873]])
    P_right = K_right @ Rt_right

    # Open the output files for writing
    with open(output_xyz_path, "w") as f_xyz, open(output_txt_path, "w") as f_txt:
        header = "# .xyz point cloud file\n# x y z r g b\n"
        f_xyz.write(header)
        f_txt.write(header)
        
        # Process each image in the input directory
        for img_file in os.listdir(input_path):
            if img_file.endswith(".jpg"):
                print(f"processing: {img_file}")
                img = cv2.imread(os.path.join(input_path, img_file))
                lines = detect_lines(img)
                points_left, points_right, left_img = process_img_sift(img, lines)
                if points_left.shape[0] > 0 and points_right.shape[0] > 0:
                    points_3D = triangulate_points(P_left, P_right, points_left, points_right)
                    # Write the 3D points to the output files with their corresponding color
                    for point, point_left in zip(points_3D, points_left):
                        color = left_img[int(point_left[1]), int(point_left[0])]
                        # Filter points based on their coordinates
                        if np.abs(point[0]) > 50 or np.abs(point[1]) > 50 or point[2] < 50 or point[2] > 200:
                            continue
                        point_str = f"{point[0]} {point[1]} {point[2]} {color[2]} {color[1]} {color[0]}\n"
                        f_xyz.write(point_str)
                        f_txt.write(point_str)
print("done")
