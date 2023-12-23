# Perspective Transformation

import cv2
import numpy as np
import matplotlib.pyplot as plt

intiPosition = [
    [[-1, -1], [-1, -1], [-1, -1], [-1, -1]],
    [[-1, -1], [-1, -1], [-1, -1], [-1, -1]],
    [[-1, -1], [-1, -1], [-1, -1], [-1, -1]],
]
targetPolygon = [400, 300]
images = ["./images/1.png", "./images/2.png", "./images/3.png"]
images = ["./images/1.png"]

DISPLAY_SIZE = (1600, 900)


def onMouse(event, x, y, flags, param):
    if event == cv2.EVENT_LBUTTONDOWN:
        img_index = param["img_index"]
        image = cv2.imread(images[img_index])
        x = int(x * (image.shape[1] / DISPLAY_SIZE[0]))
        y = int(y * (image.shape[0] / DISPLAY_SIZE[1]))
        for i in range(4):
            if intiPosition[img_index][i][0] == -1:
                intiPosition[img_index][i] = [x, y]
                break
        else:
            print("Polygon is rest")
            intiPosition[img_index] = [[-1, -1], [-1, -1], [-1, -1], [-1, -1]]


def main():
    for img_index, img in enumerate(images):
        image_in = cv2.imread(img)
        cv2.namedWindow("image")
        cv2.setMouseCallback("image", onMouse, {"img_index": img_index})
        while True:
            image = image_in.copy()
            # Draw the points
            for i in range(4):
                if intiPosition[img_index][i][0] == -1:
                    break
                cv2.circle(image, tuple(intiPosition[img_index][i]), 15, (0, 0, 255), -1)
            else:
                # Draw the polygon
                pts = np.array(intiPosition[img_index], np.int32)
                pts = pts.reshape((-1, 1, 2))
                cv2.polylines(image, [pts], True, (0, 255, 255), 3)
            image_out = cv2.resize(image, DISPLAY_SIZE)

            # put text
            cv2.putText(
                image_out,
                "Click 4 points to make a polygon",
                (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                (0, 255, 0),
                2,
            )
            cv2.putText(
                image_out,
                "Press 'c' to confirm",
                (10, 60),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                (0, 255, 0),
                2,
            )
            cv2.putText(
                image_out,
                "Press 'r' to reset",
                (10, 90),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                (0, 255, 0),
                2,
            )
            cv2.putText(
                image_out,
                "Press 'q' to quit",
                (10, 120),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
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
                return
    # print(intiPosition)
    # Perspective Transformation
    for img_index, img in enumerate(images):
        image_in = cv2.imread(img)
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
        M = cv2.getPerspectiveTransform(pts1, pts2)


        # warpPerspective without clipping
        image_out = cv2.warpPerspective(image_in, M, (targetPolygon[0], targetPolygon[1] + 300))
        cv2.imshow("image", image_out)
        cv2.waitKey(0)

    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
