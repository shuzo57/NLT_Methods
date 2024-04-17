import os
import sys
import cv2
import json

class ImageApp:
    def __init__(self, img_dir: str, resize_height: int, save_dir: str=os.getcwd()):
        self.img_dir = img_dir
        self.save_dir = save_dir
        self.resize_height = resize_height
        self.coordinates_data = {}
        self.coords = []
        self.rate = None
        self.window_name = 'image'  # ウィンドウ名をクラス変数として定義

    def onMouse(self, event, x, y, flags, params):
        if event == cv2.EVENT_LBUTTONDOWN:
            org_x, org_y = int(x / self.rate), int(y / self.rate)
            self.coords.append((org_x, org_y))
            print(self.coords)

    def getResizeRate(self, height):
        return self.resize_height / height

    def resize(self, img):
        return cv2.resize(img, None, fx=self.rate, fy=self.rate)

    def run(self):
        img_list = os.listdir(self.img_dir)
        cv2.namedWindow(self.window_name)  # ウィンドウを事前に名前付け
        cv2.moveWindow(self.window_name, 100, 100)  # ウィンドウの表示位置を設定
        for img_name in img_list:
            print(f"Processing {img_name}")
            img_path = os.path.join(self.img_dir, img_name)
            img = cv2.imread(img_path)
            if img is None:
                continue
            w, h = img.shape[:2]
            self.rate = self.getResizeRate(h)
            img = self.resize(img)
            cv2.imshow(self.window_name, img)
            cv2.setMouseCallback(self.window_name, self.onMouse)

            while True:
                key = cv2.waitKey(0) & 0xFF
                if key == ord('q'):
                    sys.exit("Quit")
                elif key == ord('c'):
                    self.coords = []
                    print(self.coords)
                elif key == ord('s'):
                    self.coordinates_data[img_name] = self.coords.copy()
                    self.coords = []
                    break

            cv2.destroyAllWindows()
        print("All coordinates data collected")
        with open('subsets.json', 'w') as f:
            json.dump(self.coordinates_data, f, indent=4)

# Usage:
if __name__ == '__main__':
    app = ImageApp(img_dir='img_bundle/left', resize_height=800, save_dir='.')
    app.run()
