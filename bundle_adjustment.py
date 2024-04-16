import os
import time
import cv2
from typing import Tuple, Dict, Union

import numpy as np
import matplotlib.pyplot as plt
plt.style.use('seaborn-v0_8-whitegrid')

from scipy.sparse import lil_matrix
from scipy.optimize import least_squares


def rotate(points, rot_vecs):
    """Rotate points by given rotation vectors. 
    Rodrigues' rotation formula is used.
    """
    theta = np.linalg.norm(rot_vecs, axis=1)[:, np.newaxis]
    with np.errstate(invalid='ignore'):
        v = rot_vecs / theta
        v = np.nan_to_num(v)
    dot = np.sum(points * v, axis=1)[:, np.newaxis]
    cos_theta = np.cos(theta)
    sin_theta = np.sin(theta)

    return cos_theta * points + sin_theta * np.cross(v, points) + dot * (1 - cos_theta) * v

def project(points, camera_params):
    """Convert 3-D points to 2-D by projecting onto images."""

    points_proj = rotate(points, camera_params[:, :3])
    points_proj += camera_params[:, 3:6]
    points_proj = -points_proj[:, :2] / points_proj[:, 2, np.newaxis]
    f = camera_params[:, 6]
    k1 = camera_params[:, 7]
    k2 = camera_params[:, 8]
    n = np.sum(points_proj**2, axis=1)
    r = 1 + k1 * n + k2 * n**2
    points_proj *= (r * f)[:, np.newaxis]
    return points_proj

def fun(params, n_cameras, n_points, camera_indices, point_indices, points_2d):
    """Compute residuals.
    `params` contains camera parameters and 3-D coordinates.
    """
    camera_params = params[:n_cameras * 9].reshape((n_cameras, 9))    
    points_3d = params[n_cameras * 9:].reshape((n_points, 3))
    points_proj = project(points_3d[point_indices], camera_params[camera_indices])
    return (points_proj - points_2d).ravel()

def bundle_adjustment_sparsity(n_cameras, n_points, camera_indices, point_indices):
    m = camera_indices.size * 2
    n = n_cameras * 9 + n_points * 3
    A = lil_matrix((m, n), dtype=int)

    i = np.arange(camera_indices.size)
    for s in range(9):
        A[2 * i, camera_indices * 9 + s] = 1
        A[2 * i + 1, camera_indices * 9 + s] = 1

    for s in range(3):
        A[2 * i, n_cameras * 9 + point_indices * 3 + s] = 1
        A[2 * i + 1, n_cameras * 9 + point_indices * 3 + s] = 1

    return A

def bundle_adjustment(
    mtx: np.ndarray,
    dist: np.ndarray,
    rvecs: np.ndarray,
    tvecs: np.ndarray,
    points_3d: np.ndarray,
    points_2d: np.ndarray,
    verbose: int = 0,
    save_fig: bool = False,
    ) -> Dict[str, Union[float, int]]:
    # 焦点距離の取得（焦点距離はx方向とy方向が同じと仮定）
    focal_length_x = mtx[0, 0]
    focal_length_y = mtx[1, 1]
    focal_length = (focal_length_x + focal_length_y) / 2

    #2つの歪み係数の取得
    dist_coeff = dist[0, :2]
    # カメラパラメータ配列の初期化
    cam = np.zeros((len(rvecs), 3, 3))

    # 各カメラのパラメータを配列に格納
    for i in range(len(rvecs)):
        cam[i, 0, :3] = rvecs[i].reshape(-1)  # 回転行列の最初の行
        cam[i, 1, :3] = tvecs[i].reshape(-1)  # 並行移動ベクトル
        cam[i, 2, 0] = focal_length  # 焦点距離
        cam[i, 2, 1:3] = dist_coeff  # 歪みパラメータ

    camera_params = cam.reshape(len(rvecs), 9)
    camera_indices = np.repeat(np.arange(33), 70)
    point_indices = np.repeat(np.arange(33), 70)
    points_2d = points_2d.reshape(2310, 2)

    n_cameras = camera_params.shape[0]
    n_points = points_3d.shape[0]

    x0 = np.hstack((camera_params.ravel(), points_3d.ravel()))
    f0 = fun(x0, n_cameras, n_points, camera_indices, point_indices, points_2d)

    if save_fig:
        # 初期状態の残差
        fig = plt.figure(figsize=(10, 4))
        plt.plot(f0)
        plt.tight_layout()
        plt.savefig("residuals.png")

    A = bundle_adjustment_sparsity(n_cameras, n_points, camera_indices, point_indices)

    t0 = time.time()
    res = least_squares(fun, x0, jac_sparsity=A, verbose=2, x_scale='jac', ftol=1e-4, method='trf',
                        args=(n_cameras, n_points, camera_indices, point_indices, points_2d))
    t1 = time.time()
    print("Optimization took {0:.0f} seconds".format(t1 - t0))

    if save_fig:
        # 最適化後の残差
        fig = plt.figure(figsize=(10, 4))
        plt.plot(res.fun)
        plt.tight_layout()
        plt.savefig("residuals_optimized.png")

    optimized_points_3d = res.x[n_cameras * 9:].reshape((n_points, 3))
    optimized_params = res.x[:n_cameras * 9].reshape((n_cameras, 9))
    optimized_rvecs = optimized_params[:, :3]
    optimized_tvecs = optimized_params[:, 3:6]

    optimized_data = {
        "mtx" : mtx,
        "dist" : dist,
        "rvecs": optimized_rvecs,
        "tvecs": optimized_tvecs,
    }
    return optimized_data

if __name__ == '__main__':
    import json
    from files import FileName
    from jsonenc import ExtendedEncoder

    calib_path = "data/calibration_data.json"
    coord_path = "data/coordinates_data.json"

    with open(calib_path) as f:
        calib_data = json.load(f)
    with open(coord_path) as f:
        coord_data = json.load(f)

    mtx = np.array(calib_data['mtx'])
    dist = np.array(calib_data['dist'])
    rvecs = np.array(calib_data['rvecs'])
    tvecs = np.array(calib_data['tvecs'])
    points_3d = np.array(coord_data['objpoints'])[0]
    points_2d = np.array(coord_data['imgpoints'])

    optimized_data = bundle_adjustment(mtx, dist, rvecs, tvecs, points_3d, points_2d, verbose=2, save_fig=True)

    with open(FileName.optimized_data, "w") as f:
        json.dump(optimized_data, f, cls=ExtendedEncoder, indent=4)