import cv2
import paddle
from paddleocr import PaddleOCR
import logging

# logging.disable(logging.DEBUG)
# logging.disable(logging.WARNING) 
ocr = PaddleOCR(use_angle_cls=True, lang="en")


def get_all_plates(imgs):
    plate = []
    if imgs is None:
        return None
    for img in imgs:
        img = cv2.resize(img, (200, 200))
        result = ocr.ocr(img, cls=True)
        try:
            # write text
            text = str(result[0][0][1][0]).replace("-", "")
            if len(text) != 7:
                continue
        

            cv2.putText(
                img,
                text,
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
        return out
    except:
        return None


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
