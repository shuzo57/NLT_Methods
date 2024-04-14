import os
import cv2
import json

import numpy as np

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

from extensions import IMAGE_EXTENSIONS
from jsonenc import ExtendedEncoder
from files import FileName

if __name__ == '__main__':
    image_dir = "img"
    calib_path = "./data/calibration_data.json"
    coord_path = "./data/coordinates_data.json"
    funda_path = "./data/fundamental_matrix_data.json"

    image_list = [f for f in os.listdir(image_dir) if f.split(".")[-1] in IMAGE_EXTENSIONS]

    with open(calib_path, "r") as f:
        calib_data = json.load(f)
    with open(coord_path, "r") as f:
        coord_data = json.load(f)
    with open(funda_path, "r") as f:
        funda_data = json.load(f)

    K = np.array(calib_data["mtx"])
    objpoints = np.array(coord_data["objpoints"])
    imgpoints = np.array(coord_data["imgpoints"])

    save_dir = "./data/chessboard_3d_relative"
    if not os.path.exists(save_dir):
        os.makedirs(save_dir, exist_ok=True)
    
    world_coordinates_data = {}
    for i in range(len(imgpoints)):
        for j in range(len(imgpoints)):
            if i <= j:
                continue
            pts1 = imgpoints[i].reshape(-1, 2).T
            pts2 = imgpoints[j].reshape(-1, 2).T

            R1, t1 = np.eye(3), np.zeros((3, 1))
            r2 = np.array(funda_data[f"{i}_{j}"]["rvec"])
            t2 = np.array(funda_data[f"{i}_{j}"]["tvec"])
            R2, _ = cv2.Rodrigues(r2)
            t2 = t2.reshape(-1, 1)

            P1 = np.dot(K, np.hstack((R1, t1)))
            P2 = np.dot(K, np.hstack((R2, t2)))

            point4D = cv2.triangulatePoints(P1, P2, pts1, pts2)
            world_coordinates = point4D[:3] / point4D[3]
            world_coordinates = world_coordinates.T

            world_coordinates_data[f"{i}_{j}"] = world_coordinates.tolist()

            save_path = os.path.join(save_dir, f"chessboard_3d_{i}_{j}.jpg")
            fig = plt.figure()
            ax = fig.add_subplot(111, projection='3d')
            ax.scatter(world_coordinates[:, 0], world_coordinates[:, 1], world_coordinates[:, 2])
            plt.savefig(save_path)
            plt.close()
    
    with open(FileName.world_coordinates_relative_data, "w") as f:
        json.dump(world_coordinates_data, f, cls=ExtendedEncoder, indent=4)
    print(f"Saved to {FileName.world_coordinates_data}")