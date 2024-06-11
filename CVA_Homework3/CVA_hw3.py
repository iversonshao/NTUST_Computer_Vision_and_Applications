import numpy as np
import cv2

pts_3d = np.array([
    [0.286055, -11.743147, 67.417709],
    [9.553491, -1.817773, 39.428276],
    [-2.264154, 6.104062, 39.440418],
    [-0.345334, 9.006487, 36.315571],
    [11.330848, 5.126206, 30.041655],
    [-8.549940, 12.928144, 21.339237],
    [10.600566, 14.927601, 11.081282],
    [1.954979, 10.586485, 0.063634],
    [-9.463017, -0.491505, 39.036213]
])

pts_2d = np.array([
    [1414, 160],
    [1220, 793],
    [1479, 831],
    [1446, 903],
    [1196, 992],
    [1622, 1270],
    [1189, 1512],
    [1396, 1704],
    [1640, 811]

])

# Compute the 3x4 projection matrix
def compute_projection_matrix(p_3D, p_2D):
    A = []
    for p_3D, p_2D in zip(p_3D, p_2D):
        X, Y, Z = p_3D
        u, v = p_2D
        A.extend([[X, Y, Z, 1, 0, 0, 0, 0, -u*X, -u*Y, -u*Z, -u],
                  [0, 0, 0, 0, X, Y, Z, 1, -v*X, -v*Y, -v*Z, -v]])
    A = np.array(A)
    _, _, V_h = np.linalg.svd(A)
    P = V_h[-1].reshape(3, 4)
    return P
#Read and convert image to RGB
def load_and_convert_image(filepath):
    image = cv2.imread(filepath)
    if image is None:
        raise FileNotFoundError(f"Image at path {filepath} not found.")
    return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
#Calculate color based on 3D point
def get_color(point, P, image):
    point_homogeneous = np.append(point, 1)
    projected = P @ point_homogeneous
    u, v, w = projected / projected[-1]
    if 0 <= u < image.shape[1] and 0 <= v < image.shape[0]:
        return image[int(v), int(u)]
    else:
        return [255, 0, 0]
#Read XYZ file, calculate color, and generate results
def process_points(filename, P, image):
    results = []
    with open(filename, 'r') as file:
        for line in file:
            data = list(map(float, line.strip().split()))
            point = np.array(data[:3])
            normal = np.array(data[3:])
            visible = normal[1] > 0
            color = get_color(point, P, image) if visible else (0, 0, 0)
            alpha = 255 if visible else 0
            results.append((point, normal, color, alpha))
    return results

if __name__ == '__main__':
    P = compute_projection_matrix(pts_3d, pts_2d)
    image = load_and_convert_image('Homework#3\\Santa.jpg')
    points_with_colors = process_points('Homework#3\\Santa.xyz', P, image)

    with open('M11215075.txt', 'w') as file:
        for point, normal, color, alpha in points_with_colors:
            file.write(f"{point[0]} {point[1]} {point[2]} {normal[0]} {normal[1]} {normal[2]} {color[0]} {color[1]} {color[2]} {alpha}\n")

    print("Output written to M11215075.txt.")