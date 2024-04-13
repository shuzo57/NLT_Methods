import os
import cv2
import json

import numpy as np

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

from extensions import IMAGE_EXTENSIONS
from jsonenc import ExtendedEncoder
from files import FileName


if __name__ == "__main__":
    image_dir = "img"
    json_path = "data/calibration_data.json"

    with open(json_path, "r") as f:
        calibration_data = json.load(f)

    image_list = [f for f in os.listdir(image_dir) if f.split(".")[-1] in IMAGE_EXTENSIONS]

    K = np.array(calibration_data["new_camera_matrix"])
    rvecs = np.array(calibration_data["rvecs"])
    tvecs = np.array(calibration_data["tvecs"])

    pattern_size = (10, 7)
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

    objp = np.zeros((pattern_size[0] * pattern_size[1], 3), np.float32)
    objp[:, :2] = np.mgrid[0:pattern_size[0], 0:pattern_size[1]].T.reshape(-1, 2)

    save_dir = "./data/chessboard_3d"
    if not os.path.exists(save_dir):
        os.makedirs(save_dir, exist_ok=True)

    world_coordinates_data = {}
    for i in range(len(image_list)):
        for j in range(len(image_list)):
            if i <= j:
                continue
            img_path1 = os.path.join(image_dir, image_list[i])
            img_path2 = os.path.join(image_dir, image_list[j])
            t1 = tvecs[i].reshape(-1)
            t2 = tvecs[j].reshape(-1)
            R1, _ = cv2.Rodrigues(rvecs[i])
            R2, _ = cv2.Rodrigues(rvecs[j])

            gray1 = cv2.imread(img_path1, cv2.IMREAD_GRAYSCALE)
            _, corners1 = cv2.findChessboardCorners(gray1, pattern_size, None)
            corners1 = cv2.cornerSubPix(gray1, corners1, (11, 11), (-1, -1), criteria)

            gray2 = cv2.imread(img_path2, cv2.IMREAD_GRAYSCALE)
            _, corners2 = cv2.findChessboardCorners(gray2, pattern_size, None)
            corners2 = cv2.cornerSubPix(gray2, corners2, (11, 11), (-1, -1), criteria)

            P1 = np.dot(K, np.hstack((R1, t1.reshape(-1, 1))))
            P2 = np.dot(K, np.hstack((R2, t2.reshape(-1, 1))))

            img_point1 = corners1.reshape(-1, 2).T
            img_point2 = corners2.reshape(-1, 2).T

            point4D = cv2.triangulatePoints(P1, P2, img_point1, img_point2)
            world_coordinates = point4D[:3] / point4D[3]
            world_coordinates = world_coordinates.T

            world_coordinates_data[f"{i}_{j}"] = world_coordinates.tolist()

            save_path = os.path.join(save_dir, f"chessboard_3d_{i}_{j}.png")
            fig = plt.figure()
            ax = fig.add_subplot(111, projection='3d')
            ax.set_xlim(-1, 13)
            ax.set_ylim(-1, 9)
            ax.set_zlim(-1, 1)
            ax.scatter(world_coordinates[:, 0], world_coordinates[:, 1], world_coordinates[:, 2])
            ax.scatter(objp[:, 0], objp[:, 1], objp[:, 2])
            plt.savefig(save_path)
            plt.close()
    
    with open(FileName.world_coordinates_data, "w") as f:
        json.dump(world_coordinates_data, f, cls=ExtendedEncoder, indent=4)
    print(f"Saved to {FileName.world_coordinates_data}")