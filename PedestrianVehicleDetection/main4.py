from collections import defaultdict
import pprint
from typing import Dict
import cv2
import numpy as np
import os
from ultralytics import YOLO
import supervision as sv
from supervision import Detections, BoxAnnotator
from supervision import ColorPalette
from supervision import Color
from supervision import Point
from matplotlib import pyplot as plt
from matplotlib.figure import Figure
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from scipy.interpolate import make_interp_spline
from supervision import Position
import datetime
import time
import threading

from walkerguard import Detector, Tracker, Window

colors = sv.ColorPalette.default()


class WalkerGuard:
    detector: Detector
    tracker = {}

    def __init__(self, model_path: str, window: Window, video_info):
        self.detector = Detector(model_path)
        print(self.detector.class_dict)
        self.tracker = {
            "行人等待區": Tracker(
                poly=[
                    np.array(
                        window.getClickPoints(4),
                    ),
                    np.array(
                        window.getClickPoints(4),
                    ),
                ],
                class_dict=self.detector.class_dict,
                video_info=video_info,
            ),
        }
        print("行人等待區 : ", " , ".join([str(x) for x in self.tracker["行人等待區"].poly]))

        self.detections_pedestrians_waiting = []
        self.detections_pedestrians_crossing = []

        self.detections_vehicle_entered = []

    def update(self, frame, detections: Detections):
        temp = self.tracker["行人等待區"].getInside(
            detections,
            frame,
            byTime=3,
            class_name="",
            labelFunc=lambda x: "Waiting" if x > 0 else "Nobody",
        )
        self.detections_pedestrians_waiting = temp[0]  # [Detections, Detections, ...]
        count = temp[1]  # [int, int, ...]
        annotated_frame = temp[2]  # visualized frame
        return annotated_frame

    def isPedestriansWaiting(self, detections: Detections):
        return True

    def isPedestriansCrossing(self, detections: Detections):
        return True


window = Window("Preview")
triangleA_annotator = sv.TriangleAnnotator()


def main():
    video_path = "./video/WIN_20231229_11_33_12_Pro.mp4"
    video_info = sv.VideoInfo.from_video_path(video_path=video_path)
    cap = cv2.VideoCapture(video_path)
    # show first frame
    first_frame = cap.read()[1]
    window.image = first_frame

    walkerGuard = WalkerGuard("./model/best.pt", window, video_info)

    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
    while cap.isOpened():
        ret, frame = cap.read()
        if ret:
            detections, results = walkerGuard.detector.detect(frame, conf=0.2, verbose=False)
            frame = triangleA_annotator.annotate(frame, detections)
            image = walkerGuard.update(frame, detections)
            window.update(image)

            if window.key == ord('s'):
                # skip 5 seconds
                cap.set(cv2.CAP_PROP_POS_FRAMES, cap.get(cv2.CAP_PROP_POS_FRAMES) + 5 * 30)


if __name__ == "__main__":
    try:
        thread = threading.Thread(target=main)
        thread.setDaemon(True)
        thread.start()
        while True:
            if window.run():
                break
    except KeyboardInterrupt:
        exit(0)
