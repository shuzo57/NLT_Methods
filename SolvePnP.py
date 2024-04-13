import os
import cv2
import json

import numpy as np

from jsonenc import ExtendedEncoder
from files import FileName

if __name__ == '__main__':
    calib_path = "./data/calibration_data.json"
    coord_path = "./data/coordinates_data.json"

    with open(calib_path, "r") as f:
        calib_data = json.load(f)
    with open(coord_path, "r") as f:
        coord_data = json.load(f)

    mtx = np.array(calib_data["mtx"])
    dist = np.array(calib_data["dist"])
    objpoints = np.array(coord_data["objpoints"])
    imgpoints = np.array(coord_data["imgpoints"])
    
    rvecs = np.zeros((len(objpoints), 3))
    tvecs = np.zeros((len(objpoints), 3))

    for i in range(len(objpoints)):
        _, rvec, tvec = cv2.solvePnP(objpoints[i], imgpoints[i], mtx, dist)
        rvecs[i] = rvec.reshape(3)
        tvecs[i] = tvec.reshape(3)
    
    data = {
        "rvecs": rvecs,
        "tvecs": tvecs,
    }

    with open(FileName.solve_pnp_data, "w") as f:
        json.dump(data, f, cls=ExtendedEncoder, indent=4)