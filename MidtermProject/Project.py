import cv2
import numpy as np
import os

# Initialize an empty list to store all XYZ values
all_points_3d = np.zeros((0, 3), dtype=float)

data_dir = 'MidtermProject\\MidtermProject\\ShadowStrip'
cube = 200
for i in range(55):
    # Load the image
    img_path = os.path.join(data_dir, f"{i:04d}.jpg")
    img = cv2.imread(img_path)
    # Convert to HSV color space
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

    # Define the range for red color
    lower_red = np.array([0, 50, 50])
    upper_red = np.array([50, 255, 255])
    # Create a mask for red color
    mask = cv2.inRange(hsv, lower_red, upper_red)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (7, 7))
    mask = cv2.dilate(mask, kernel) # Dilate
    mask = cv2.erode(mask, kernel)  # Erode
    # Find contours
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Initialize coordinates for 4 corners
    left_top = right_top = right_bottom = left_bottom = None

    # Iterate through contours
    for contour in contours:
        # Calculate the centroid of the contour
        M = cv2.moments(contour)
        if M["m00"] != 0:
            cX = int(M["m10"] / M["m00"])
            cY = int(M["m01"] / M["m00"])

            # Update coordinates of the four corners
            if left_top is None or (cX + cY) < (left_top[0] + left_top[1]):
                left_top = (cX, cY)
            if left_bottom is None or (cX - cY) > (left_bottom[0] - left_bottom[1]):
                left_bottom = (cX, cY)
            if right_top is None or (cX - cY) < (right_top[0] - right_top[1]):
                right_top = (cX, cY)
            if right_bottom is None or (cX + cY) > (right_bottom[0] + right_bottom[1]):
                right_bottom = (cX, cY)

    # Perspective transformation
    pixel2D = np.float32([left_top, right_top, right_bottom, left_bottom])
    real3D = np.float32([[0, 0], [0, cube], [cube, cube], [cube, 0]])
    matrix = cv2.getPerspectiveTransform(pixel2D, real3D)
    mask = cv2.warpPerspective(mask, matrix, (200, 200))
    

    # Find all points on the right boundary of the white area
    wp = np.where(mask == 255)
    r_boundary = []

    for points in range(len(wp[0])):
        if wp[1][points] == max(wp[1][np.where(wp[0] == wp[0][points])]):
            r_boundary.append((wp[0][points], wp[1][points]))

    # Mark the points on the right boundary
    for point in r_boundary:
        x = i - 27.5 #27.5 is the middle of the 55 images
        y = 100 - point[1] #reverse the y-axis
        z = 100 - point[0] #reverse the x-axis
        all_points_3d = np.append(all_points_3d, [[x, y, z]], axis=0)

# Save XYZ values to a file

np.savetxt('M11215075.xyz', all_points_3d, fmt='%.8f')
