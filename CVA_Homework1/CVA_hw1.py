import cv2
import numpy as np

# read camera parameters
def rd_camera_par(file_path):
    K1, RT1, K2, RT2 = [], [], [], []
    with open(file_path, "r") as f:
        data = f.readlines()
    Matrix = None
    for line in data:
        if line.startswith("Cam1_K"):
            Matrix = 'K1'
        elif line.startswith("Cam1_RT"):
            Matrix = 'RT1'
        elif line.startswith("Cam2_K"):
            Matrix = 'K2'
        elif line.startswith("Cam2_RT"):
            Matrix = 'RT2'

        if Matrix == 'K1' and line.strip() != 'Cam1_K':
            K1.extend([float(i) for i in line.replace('[', '').replace(']', '').replace('\n', '').split()])
        elif Matrix == 'RT1' and line.strip() != 'Cam1_RT':
            RT1.extend([float(i) for i in line.replace('[', '').replace(']', '').replace('\n', '').split()])
        elif Matrix == 'K2' and line.strip() != 'Cam2_K':
            K2.extend([float(i) for i in line.replace('[', '').replace(']', '').replace('\n', '').split()])
        elif Matrix == 'RT2' and line.strip() != 'Cam2_RT':
            RT2.extend([float(i) for i in line.replace('[', '').replace(']', '').replace('\n', '').split()])
    
    K1 = np.array(K1).reshape(3, 3)
    RT1 = np.array(RT1).reshape(3, 4)
    K2 = np.array(K2).reshape(3, 3)
    RT2 = np.array(RT2).reshape(3, 4)

    return K1, RT1, K2, RT2
#read 3D trajectory
def rd_3D_Trajectory(file_path):
    Trajectory = []
    with open(file_path, "r") as f:
        data = f.readlines()
        for line in data:
            X_3D = line.replace('\n', '').split()
            X_3D.append(1)  # Assuming 1 is added for homogeneous coordinates
            Trajectory.append([float(i) for i in X_3D])
    return np.array(Trajectory)

# x = K[R|T]X_3D
def camera_model(K_src, RT_src, X_3D_src):
    Trajectory_Img = []
    for i in X_3D_src:
        i = np.array(i)
        Mul = K_src @ RT_src @ i
        # Normalization z = 1
        Mul = Mul / Mul[2]
        # Remove z
        Mul = np.delete(Mul, (2), axis=0)
        Trajectory_Img.append(Mul)
    return Trajectory_Img


if __name__ == '__main__':

    #input:soccer
    img1 = cv2.imread("Homework#1/SceneFromCamera1.jpg")
    img2 = cv2.imread("Homework#1/SceneFromCamera2.jpg")

    # read camera parameters
    K1, RT1, K2, RT2 = rd_camera_par("Homework#1/CameraParameter.txt")
    # read 3D trajectory
    X = rd_3D_Trajectory("Homework#1/3D_Trajectory.xyz")

    Trajectory_Img1 = camera_model(K1, RT1, X)
    Trajectory_Img2 = camera_model(K2, RT2, X)

    # # draw 3D trajectory
    for i in Trajectory_Img1:
        img1_1 = cv2.circle(img1, (int(i[0]), int(i[1])), 3, (0, 0, 255), -1)
    cv2.polylines(img1, [np.int32(Trajectory_Img1)], isClosed=False, color=(0, 0, 255), thickness=2)

    for i in Trajectory_Img2:
        img2_1 = cv2.circle(img2, (int(i[0]), int(i[1])), 3, (0, 60, 255), -1)
    cv2.polylines(img2, [np.int32(Trajectory_Img2)], isClosed=False, color=(0, 0, 255), thickness=2)

    cv2.imshow("img1", img1_1)
    cv2.waitKey(0)
    cv2.imshow("img2", img2_1)
    cv2.waitKey(0)
    cv2.imwrite("M11215075_1.jpg", img1_1)
    cv2.imwrite("M11215075_2.jpg", img2_1)
    cv2.destroyAllWindows()


