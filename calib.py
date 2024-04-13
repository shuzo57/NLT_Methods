import os
import cv2
import numpy as np
from typing import Tuple, Dict, Union

from extensions import IMAGE_EXTENSIONS
from jsonenc import ExtendedEncoder
from files import FileName

def intrinsics_calibration(
    image_dir: str,
    pattern_size: Tuple[int, int] = (10, 7),
    result_dir: str = None,
    save_coords: bool = False,
    ) -> Dict[str, Union[float, int]]:
    
    if result_dir is not None and not os.path.exists(result_dir):
        os.makedirs(result_dir, exist_ok=True)

    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

    objp = np.zeros((pattern_size[0] * pattern_size[1], 3), np.float32)
    objp[:, :2] = np.mgrid[0:pattern_size[0], 0:pattern_size[1]].T.reshape(-1, 2)

    objpoints = []
    imgpoints = []
    calibrated_cnt = 0
    h, w = None, None

    for file in os.listdir(image_dir):
        ex = file.split('.')[-1]
        if ex not in IMAGE_EXTENSIONS:
            continue
        print(f'\rProcessing: {calibrated_cnt}', end='')

        img_path = os.path.join(image_dir, file)
        img = cv2.imread(img_path)
        if img is None:
            continue
        if h is None or w is None:
            h, w = img.shape[:2]

        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        ret, corners = cv2.findChessboardCorners(gray, pattern_size, None)
        
        if not ret:
            continue

        objpoints.append(objp)
        corners = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)
        imgpoints.append(corners)

        if result_dir is not None:
            cv2.drawChessboardCorners(img, pattern_size, corners, ret)
            save_path = os.path.join(result_dir, f'calib_{file}.jpg')
            cv2.imwrite(save_path, img)

        calibrated_cnt += 1
    
    print(f'\nCalibrated {calibrated_cnt} images')

    if len(objpoints) == 0 or len(imgpoints) == 0:
        raise ValueError('No calibration data found')
    
    ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, (w, h), None, None)
    new_camera_matrix, roi = cv2.getOptimalNewCameraMatrix(mtx, dist, (w, h), 1, (w, h))

    calibration_data = dict(
        mtx = mtx,
        dist = dist,
        rvecs = rvecs,
        tvecs = tvecs,
        new_camera_matrix = new_camera_matrix,
        roi = roi,
    )

    if save_coords:
        coordinates_data = dict(
            objpoints = objpoints,
            imgpoints = imgpoints,
            pattern_size = pattern_size,
        )
        save_path = os.path.join(result_dir, FileName.coordinates_data)
        with open(save_path, 'w') as f:
            json.dump(coordinates_data, f, indent=4, cls=ExtendedEncoder)
        print('Calibration data saved to', save_path)
    
    return calibration_data


if __name__ == '__main__':
    from files import FileName
    import json

    image_dir = "img"
    calibration_data = intrinsics_calibration(image_dir, result_dir='data/calib_result', save_coords=True)
    save_path = os.path.join(os.getcwd(), FileName.calibration_data)

    with open(save_path, 'w') as f:
        json.dump(calibration_data, f, indent=4, cls=ExtendedEncoder)

    print('Calibration data saved to', save_path)