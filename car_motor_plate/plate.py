import cv2
import paddle
from paddleocr import PaddleOCR

ocr = PaddleOCR(use_angle_cls=True, lang="en")


def get_all_plates(imgs):
    plate = []
    print("\n\n\n\n\n")
    for img in imgs:
        img = cv2.resize(img, (200, 200))
        result = ocr.ocr(img, cls=True)
        print(result)
        # write text
        try:
            cv2.putText(
                img,
                result[0][0][1][0],
                (10, 40),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                (255, 0, 0),
                2,
                cv2.LINE_AA,
            )

            plate.append(img)
        except:
            pass
    try:
        out = cv2.hconcat(plate)
        cv2.imshow("plate", out)
        cv2.waitKey(1)
        return out
    except:
        pass


[
    [
        [
            [[18.0, 78.0], [184.0, 78.0], [184.0, 146.0], [18.0, 146.0]],
            ("NQF-6695", 0.9348689317703247),
        ]
    ]
]
# # read image
# img = cv2.imread("plate.jpg")
# get_all_plates([img])
