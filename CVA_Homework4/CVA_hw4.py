import numpy as np
import cv2
import math

np.set_printoptions(suppress=True, precision=6)

# Load the image and convert it to grayscale
image_path = "Homework#4\\imgToCalib.png"
img = cv2.imread(image_path)
if img is None:
    raise FileNotFoundError(f"Image at path {image_path} not found.")
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# 3D coordinates of points (flattened to 2D array)
objp = np.array([
    [8, 0, 6], [8, 6, 6], [8, 6, 0], [8, 0, 0],  # First square
    [0, 0, 6], [0, 6, 6], [8, 6, 6], [8, 0, 6],  # Second square
    [0, 6, 0], [8, 6, 0], [8, 6, 6], [0, 6, 6]   # Third square
], dtype=np.float32)

# 2D pixel coordinates (flattened to 2D array)
imgp = np.array([
    [358, 159], [631, 350], [687, 1022], [436, 711],  # First square
    [1050, 57], [1438, 177], [631, 350], [358, 159],  # Second square
    [1375, 737], [687, 1022], [631, 350], [1438, 177]  # Third square
], dtype=np.float32)

objpoints = [objp]
imgpoints = [imgp]

# Estimate initial intrinsic matrix
# Image size
image_size = (1920, 1080)
# Focal length
fx = fy = max(image_size) / math.pi
# Principal point
cx = image_size[0] / 2
cy = image_size[1] / 2
K = np.array([[fx, 0, cx], 
              [0, fy, cy], 
              [0, 0, 1]], dtype=np.float32)

# Zero distortion coefficients
dist_coeffs = np.zeros((4, 1), dtype=np.float32)

# Estimate camera parameters
ret, K, dist, rvecs, tvecs = cv2.calibrateCamera(
    objpoints, imgpoints, image_size, K, dist_coeffs, flags=cv2.fisheye.CALIB_RECOMPUTE_EXTRINSIC+cv2.fisheye.CALIB_CHECK_COND + \
    cv2.fisheye.CALIB_FIX_SKEW+cv2.fisheye.CALIB_USE_INTRINSIC_GUESS
)
# mtx = np.round(mtx,1)
# a) Output intrinsic parameter matrix K
print("a) Intrinsic parameter matrix K:")
print(K)

# b) Output extrinsic parameter matrix RT
rotation_matrix = cv2.Rodrigues(rvecs[0])[0]  # Convert rotation vector to rotation matrix
extrinsic_matrix = np.hstack((rotation_matrix, tvecs[0]))  # Combine rotation matrix and translation vector into extrinsic matrix
print("\nb) Extrinsic parameter matrix RT:")
print(extrinsic_matrix)

# c) Calculate distance from camera to origin
camera_position = -np.matmul(rotation_matrix.T, tvecs[0])
camera_distance = np.linalg.norm(camera_position)
print(f"\nc) Distance from camera to origin: {camera_distance:.1f} meters")