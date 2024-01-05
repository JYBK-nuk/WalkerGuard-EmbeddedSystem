# Perspective Transformation

import cv2
import numpy as np
import matplotlib.pyplot as plt
import os
intiPosition = [
    [[-1, -1], [-1, -1], [-1, -1], [-1, -1]],
    [[-1, -1], [-1, -1], [-1, -1], [-1, -1]],
    [[-1, -1], [-1, -1], [-1, -1], [-1, -1]],
]
targetPolygon = [400, 300]
images = ["./images/4.png", "./images/5.png", "./images/6.png"]

image_in = [cv2.imread(img) for img in images]
image_in = [cv2.resize(img, (800, 600)) for img in image_in]


imagesMatrix = [None for i in range(len(images))]

# 轉換後地圖底圖
TransformedMap = None

DISPLAY_SIZE = (600, 400)


def onMouse(event, x, y, flags, param):
    global image_in, intiPosition
    if event == cv2.EVENT_LBUTTONDOWN:
        img_index = param["img_index"]
        image = image_in[img_index].copy()
        # 還原成原圖x,y (因為傳入的是resize後的圖片，滑鼠點擊的位置)
        x = int(x * (image.shape[1] / DISPLAY_SIZE[0]))
        y = int(y * (image.shape[0] / DISPLAY_SIZE[1]))
        for i in range(4):
            if intiPosition[img_index][i][0] == -1:
                intiPosition[img_index][i] = [x, y]
                break
        else:
            print("Polygon is rest")
            intiPosition[img_index] = [[-1, -1], [-1, -1], [-1, -1], [-1, -1]]


### 讓每張圖都選擇四個點
def inputPolygonPerImage():
    global image_in, intiPosition, imagesMatrix, TransformedMap
    for img_index, img in enumerate(images):
        cv2.namedWindow("image")
        cv2.setMouseCallback("image", onMouse, {"img_index": img_index})
        while True:
            image = image_in[img_index].copy()
            # Draw the points
            for i in range(4):
                if intiPosition[img_index][i][0] == -1:
                    break
                cv2.circle(image, tuple(intiPosition[img_index][i]), 3, (0, 0, 255), -1)
            else:
                # Draw the polygon
                pts = np.array(intiPosition[img_index], np.int32)
                pts = pts.reshape((-1, 1, 2))
                cv2.polylines(image, [pts], True, (0, 255, 255), 2)
            image_out = cv2.resize(image, DISPLAY_SIZE)

            # put text
            cv2.putText(
                image_out,
                "Click 4 points to make a polygon",
                (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (0, 255, 0),
                2,
            )
            cv2.putText(
                image_out,
                "Press 'c' to confirm",
                (10, 50),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (0, 255, 0),
                2,
            )
            cv2.putText(
                image_out,
                "Press 'r' to reset",
                (10, 70),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (0, 255, 0),
                2,
            )
            cv2.putText(
                image_out,
                "Press 'q' to quit",
                (10, 90),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (0, 255, 0),
                2,
            )
            cv2.imshow("image", image_out)
            key = cv2.waitKey(1) & 0xFF
            if key == ord("c"):
                # confirm
                if intiPosition[img_index][3][0] == -1:
                    print("Polygon is not complete")
                else:
                    print("Polygon is complete")
                    break
            elif key == ord("r"):
                intiPosition[img_index] = [[-1, -1], [-1, -1], [-1, -1], [-1, -1]]
                print("Polygon is rest")
            elif key == ord("q"):
                cv2.destroyAllWindows()
                exit()


# 計算轉換矩陣
def getAllMatrixAndTransformedImage():
    global image_in, intiPosition, imagesMatrix, TransformedMap
    for img_index, img in enumerate(images):
        image = image_in[img_index].copy()
        pts1 = np.float32(intiPosition[img_index])
        # 左上、左下、右下、右上 (左上為0,0)
        pts2 = np.float32(
            [
                [0, 0],
                [0, targetPolygon[1]],
                [targetPolygon[0], targetPolygon[1]],
                [targetPolygon[0], 0],
            ]
        )
        # 計算轉換矩陣
        M = cv2.getPerspectiveTransform(pts1, pts2)
        imagesMatrix[img_index] = M

        # 等等展示用的地圖
        global TransformedMap
        dst = cv2.warpPerspective(image, M, (targetPolygon[0], targetPolygon[1]))  # 轉換圖片到目標透視圖
        TransformedMap = dst
    cv2.destroyAllWindows()


# 每張圖片對應的點
points = [[-1, -1] for _ in range(len(images))]
# 轉換後的圖片上的點
plt_point = (0, 0)


# 處理滑鼠點擊任意一張圖片時，其他圖片的點也會跟著轉換到對應位置
def Transform(event, x, y, flags, param):
    global plt_point, points, imagesMatrix, image_in, DISPLAY_SIZE
    if event == cv2.EVENT_LBUTTONDOWN:
        img_index = param["img_index"]
        image = image_in[img_index].copy()
        # 還原成原圖x,y (因為傳入的是resize後的圖片，滑鼠點擊的位置)
        x = int(x * (image.shape[1] / DISPLAY_SIZE[0]))
        y = int(y * (image.shape[0] / DISPLAY_SIZE[1]))
        points[img_index] = [x, y]
        m = imagesMatrix[img_index]
        # 轉換到Map上的點
        transformed_x, transformed_y = cv2.perspectiveTransform(
            np.array([[[x, y]]], dtype=np.float32), m
        )[0][0]
        # 更新Map上的點
        plt_point = [int(transformed_x), int(transformed_y)]
        for i in range(len(images)):
            if i != img_index:
                m = imagesMatrix[i]
                # 再從Map上的點轉換回來 (inverse Matrix)
                x, y = cv2.perspectiveTransform(
                    np.array([[[transformed_x, transformed_y]]], dtype=np.float32),
                    np.linalg.inv(m),
                )[0][0]
                # 更新其他圖片上對應的點
                points[i] = [int(x), int(y)]


def main():
    global TransformedMap, points, plt_point, images, image_in, DISPLAY_SIZE
    for img_index, img in enumerate(images):
        cv2.namedWindow(F"image{img_index}")
        cv2.setMouseCallback(F"image{img_index}", Transform, {"img_index": img_index})
    while True:
        TransformedMap_copy = TransformedMap.copy()
        for img_index, img in enumerate(images):
            image = image_in[img_index].copy()
            # Draw the points
            cv2.circle(image, tuple(points[img_index]), 3, (0, 0, 255), -1)
            image_out = cv2.resize(image, DISPLAY_SIZE)
            cv2.imshow(F"image{img_index}", image_out)

        cv2.circle(TransformedMap_copy, plt_point, 3, (0, 0, 255), -1)
        cv2.imshow("Map", TransformedMap_copy)
        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            print("Quit")
            return


if __name__ == "__main__":
    inputPolygonPerImage()
    getAllMatrixAndTransformedImage()
    main()
