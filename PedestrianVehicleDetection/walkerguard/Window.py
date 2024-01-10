import cv2
import numpy as np
import time
import colorama
from PIL import Image, ImageDraw, ImageFont
import requests

colorama.init()
font = ImageFont.truetype('msjhbd.ttc', 40)


class Window:
    def __init__(self, window_name, window_size: tuple = (1280, 720)):
        self.window_name = window_name
        self.signal = "行人紅燈"
        self.image = np.zeros((window_size[1], window_size[0], 3), np.uint8)
        self.text = []
        self.text_reset_time = None
        self.text_format = {
            "color": (0, 0, 255),
        }

        cv2.namedWindow(self.window_name, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(self.window_name, window_size[0], window_size[1])
        cv2.setMouseCallback(self.window_name, self.__mouse_callback)
        self.__clickPoints = []
        self.__gettingClickPoints = False
        self.key = None

    def SignalChange(self):
        ls = ["行人紅燈", "行人綠燈", "無號誌"]
        self.signal = ls[(ls.index(self.signal) + 1) % len(ls)]

    def __drawText(self, image: np.ndarray):
        y = 50
        imgPil = Image.fromarray(image)
        draw = ImageDraw.Draw(imgPil)
        draw.text((0, 0), self.signal, fill=(0, 255, 0), font=font)
        for text in self.text:
            draw.text(
                (0, y),
                text,
                fill=self.text_format["color"],
                font=font,
                stroke_width=1,
                stroke_fill=(0, 0, 0),
            )
            y += 50
        image = np.array(imgPil)
        return image

    def update(self, image: np.ndarray):
        self.image = image
        image = image.copy()
        if self.text_reset_time is not None:
            if time.time() > self.text_reset_time:
                self.text = []
                self.text_reset_time = None
        image = self.__drawText(image)
        # draw points and polygons
        if len(self.__clickPoints) > 0:
            for point in self.__clickPoints:
                image = cv2.circle(image, point, 5, (0, 255, 0), -1)
            if len(self.__clickPoints) > 1:
                image = cv2.polylines(image, [np.array(self.__clickPoints)], True, (0, 255, 0), 2)

        cv2.imshow(self.window_name, image)

    def showText(
        self,
        text: list[str],
        color: tuple = (255, 150, 150),
        duration: int = 3,
    ):
        # print with timestamp and color
        print(
            colorama.Fore.GREEN
            + time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
            + colorama.Style.RESET_ALL
            + "\n"
            + str(text)
        )

        self.text = text
        self.text_reset_time = time.time() + duration
        self.text_format = {
            "color": color,
        }

    def __mouse_callback(self, event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            if self.__gettingClickPoints:
                self.__clickPoints.append((x, y))
            else:
                self.SignalChange()
        elif event == cv2.EVENT_RBUTTONDOWN:
            requests.post("http://127.0.0.1/violate", json={"x": x, "y": y})
            print("Test Violate!")

    def getClickPoints(self, num: int = 4):
        self.__gettingClickPoints = True
        self.__clickPoints = []
        while True:
            self.update(self.image)
            if self.key == ord('r'):
                self.__clickPoints = []
                print("Reset!")
            if len(self.__clickPoints) == num:
                if self.key == ord('c'):
                    print(F"Confirm click points : {self.__clickPoints}")
                    break
            elif len(self.__clickPoints) > num:
                self.__clickPoints = []
        clickPoints = self.__clickPoints
        self.__clickPoints = []
        self.__gettingClickPoints = False
        return clickPoints

    def run(self):
        self.update(self.image)
        self.key = cv2.waitKey(1) & 0xFF
        if self.key == ord('q'):
            return True
