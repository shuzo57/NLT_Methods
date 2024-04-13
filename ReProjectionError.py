import os
import cv2
import json

import numpy as np

from extensions import IMAGE_EXTENSIONS


if __name__ == '__main__':
    image_dir = "img"
    calib_path = "./data/calibration_data.json"
    coord_path = "./data/coordinates_data.json"

    image_list = [f for f in os.listdir(image_dir) if f.split(".")[-1] in IMAGE_EXTENSIONS]

    with open(calib_path, "r") as f:
        calib_data = json.load(f)
    with open(coord_path, "r") as f:
        coord_data = json.load(f)

    mtx = np.array(calib_data["mtx"])
    dist = np.array(calib_data["dist"])
    rvecs = np.array(calib_data["rvecs"])
    tvecs = np.array(calib_data["tvecs"])
    objpoints = np.array(coord_data["objpoints"])
    imgpoints = np.array(coord_data["imgpoints"])

    save_dir = "./data/reprojection_error"
    if not os.path.exists(save_dir):
        os.makedirs(save_dir, exist_ok=True)

    mean_error = 0
    for i in range(len(objpoints)):
        imgpoints2, _ = cv2.projectPoints(objpoints[i], rvecs[i], tvecs[i], mtx, dist)
        error = cv2.norm(imgpoints[i], imgpoints2, cv2.NORM_L2) / len(imgpoints2)
        mean_error += error

        img = cv2.imread(os.path.join(image_dir, image_list[i]))
        for x, y in imgpoints2.reshape(-1, 2):
            cv2.circle(img, (int(x), int(y)), 5, (0, 255, 0), -1)
        save_path = os.path.join(save_dir, f"reproj_{image_list[i]}.jpg")
        cv2.imwrite(save_path, img)

    total_error = mean_error/len(objpoints)

    print(f"total error: {total_error:.3f}")