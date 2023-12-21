import cv2
import paddle
gpu_available  = paddle.device.is_compiled_with_cuda()
from paddleocr import PaddleOCR

ocr = PaddleOCR(use_angle_cls=True, lang='en')

def get_all_plates(imgs):
    plate = []
    print("\n\n\n\n\n")
    for img in imgs:
        img = cv2.resize(img, (200, 200))
        plate.append(img)
        result = ocr.ocr(img, cls=True)
        print(result)
    out = cv2.hconcat(plate)
    cv2.imshow("plate", out)
    cv2.waitKey(0)


# # read image
# img = cv2.imread("plate.jpg")
# get_all_plates([img])