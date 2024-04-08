# Purpose: カメラの内部パラメータを求める

#解説1
import cv2
import os
import numpy as np
import matplotlib.pyplot as plt

#解説2
# 終了基準
criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

# 物体点を準備する、例えば (0,0,0), (1,0,0), (2,0,0) ....,(9,12,0)
objp = np.zeros((7*10,3), np.float32)
objp[:,:2] = np.mgrid[0:10,0:7].T.reshape(-1,2)

# 全ての画像から物体点と画像点を保存するための配列
objpoints = [] # 実世界空間での3D点
imgpoints = [] # 画像平面での2D点

#解説3
images = os.listdir('img')

for fname in images:
    print(f'処理中の画像: {fname}')
    img_path = os.path.join('img', fname)
    img = cv2.imread(img_path)
    if img is None:
        print(f'画像の読み込みに失敗しました {fname}')
        continue
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)


#解説4
    # チェスボードのコーナーを見つける
    ret, corners = cv2.findChessboardCorners(gray, (10,7), None)

    # 見つかった場合、物体点と画像点（これらを精緻化した後）を追加する
    if ret == True:
        objpoints.append(objp)

        corners2 = cv2.cornerSubPix(gray,corners, (11,11), (-1,-1), criteria)
        imgpoints.append(corners2)
        # コーナーを描画して表示する
        img = cv2.drawChessboardCorners(img, (10,7), corners2, ret)

#解説5
        plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        plt.show()
    else:
        print(f'チェスボードのコーナーが見つかりませんでした {fname}')

if len(objpoints) > 0 and len(imgpoints) > 0:
    ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, gray.shape[::-1], None, None)

    print('カメラ行列 : \n')
    print(mtx)
    print('歪み : \n')
    print(dist)
    print('回転ベクトル : \n')
    print(rvecs)
    print('平行移動ベクトル : \n')
    print(tvecs)
else:
    print('画像点が不足しているため、カメラのキャリブレーションに失敗しました')
