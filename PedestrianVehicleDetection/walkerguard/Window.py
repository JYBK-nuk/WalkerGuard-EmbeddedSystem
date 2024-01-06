import cv2
import numpy as np
import time
import colorama

colorama.init()


class Window:
    def __init__(self, window_name, window_size: tuple = (1600, 900)):
        self.window_name = window_name
        self.image = np.zeros((window_size[1], window_size[0], 3), np.uint8)
        self.text = ""
        self.text_reset_time = None
        self.text_format = {
            "position": (0, 0),
            "color": (0, 0, 255),
            "font_scale": 1,
        }
        cv2.namedWindow(self.window_name, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(self.window_name, window_size[0], window_size[1])
        self.__clickPoints = []
        self.key = None

    def update(self, image: np.ndarray):
        self.image = image
        image = image.copy()
        if self.text_reset_time is not None:
            if time.time() > self.text_reset_time:
                self.text = ""
                self.text_reset_time = None

        if self.text != "":
            image = cv2.putText(
                image,
                self.text,
                self.text_format["position"],
                cv2.FONT_HERSHEY_SIMPLEX,
                self.text_format["font_scale"],
                self.text_format["color"],
                2,
            )
        # draw points and polygons
        if len(self.__clickPoints) > 0:
            for point in self.__clickPoints:
                image = cv2.circle(image, point, 5, (0, 255, 0), -1)
            if len(self.__clickPoints) > 1:
                image = cv2.polylines(image, [np.array(self.__clickPoints)], True, (0, 255, 0), 2)

        cv2.imshow(self.window_name, image)

    def showText(
        self,
        text: str,
        position: tuple = (0, 0),
        color: tuple = (0, 0, 255),
        font_scale: float = 1,
        duration: int = 3,
    ):
        # print with timestamp and color
        print(
            colorama.Fore.GREEN
            + time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
            + colorama.Style.RESET_ALL
            + " "
            + text
        )

        self.text = text
        self.text_reset_time = time.time() + duration
        self.text_format = {
            "position": position,
            "color": color,
            "font_scale": font_scale,
        }

    def __mouse_callback(self, event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            self.__clickPoints.append((x, y))

    def __mouse_nothing(self, event, x, y, flags, param):
        pass

    def getClickPoints(self, num: int = 4):
        self.__clickPoints = []
        cv2.setMouseCallback(self.window_name, self.__mouse_callback)
        while True:
            self.update(self.image)
            if self.key == ord('r'):
                self.__clickPoints = []
                print("reset")
            if len(self.__clickPoints) == num:
                if self.key == ord('c'):
                    print("confirm")
                    break
            elif len(self.__clickPoints) > num:
                self.__clickPoints = []
        clickPoints = self.__clickPoints
        self.__clickPoints = []
        cv2.setMouseCallback(self.window_name, self.__mouse_nothing)
        return clickPoints

    def run(self):
        self.update(self.image)
        self.key = cv2.waitKey(1) & 0xFF
        if self.key == ord('q'):
            return False
