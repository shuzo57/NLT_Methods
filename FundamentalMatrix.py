import os
import cv2
import json

import numpy as np
from scipy.spatial.transform import Rotation

from jsonenc import ExtendedEncoder
from files import FileName


def compute_extrinsic_parameters(
    pts1: np.ndarray, pts2: np.ndarray, K: np.ndarray
):
    F, mask = cv2.findFundamentalMat(pts1, pts2, cv2.FM_RANSAC)

    pts1 = pts1[mask.ravel() == 1]
    pts2 = pts2[mask.ravel() == 1]

    E = K.T.dot(F).dot(K)

    U, S, Vt = np.linalg.svd(E)
    W = np.array([[0, -1, 0], [1, 0, 0], [0, 0, 1]])
    R = U.dot(W).dot(Vt)
    t = U[:, 2]

    return R, t, F


if __name__ == '__main__':
    calib_path = "./data/calibration_data.json"
    coord_path = "./data/coordinates_data.json"

    with open(calib_path, "r") as f:
        calib_data = json.load(f)
    with open(coord_path, "r") as f:
        coord_data = json.load(f)

    K = np.array(calib_data["mtx"])
    imgpoints = np.array(coord_data["imgpoints"])

    fundamental_matrix_data = {}
    for i in range(len(imgpoints)):
        for j in range(len(imgpoints)):
            if i <= j:
                continue
            pts1 = imgpoints[i].reshape(-1, 2)
            pts2 = imgpoints[j].reshape(-1, 2)

            R, tvec, F = compute_extrinsic_parameters(pts1, pts2, K)
            rvec = Rotation.from_matrix(R).as_rotvec()

            data = {
                "rvec": rvec,
                "tvec": tvec,
                "F": F,
            }
            fundamental_matrix_data[f"{i}_{j}"] = data
    
    with open(FileName.fundamental_matrix_data, "w") as f:
        json.dump(fundamental_matrix_data, f, cls=ExtendedEncoder, indent=4)