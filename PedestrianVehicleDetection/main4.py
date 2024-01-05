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
    violate_history = {}

    def __init__(self, model_path: str, window: Window, video_info):
        self.detector = Detector(model_path)
        print(self.detector.class_dict)
        self.tracker = {
            "行人等待區": Tracker(
                poly=[
                    np.array(
                        window.getClickPoints(4),
                    ),
                ],
                class_dict=self.detector.class_dict,
                video_info=video_info,
            ),
            "斑馬線": Tracker(
                poly=[
                    np.array(
                        window.getClickPoints(4),
                    ),
                ],
                class_dict=self.detector.class_dict,
                video_info=video_info,
            ),
        }

        self.detections_pedestrians_waiting = []
        self.detections_pedestrians_crossing = []
        self.detections_vehicle_entered = []

    def update(self, annotated_frame, detections: Detections):
        temp = self.tracker["行人等待區"].getInside(
            detections,
            annotated_frame,
            byTime=2,
            class_name=["people", "pedestrian"],
            labelFunc=lambda x: "Waiting" if x > 0 else "Nobody",
        )
        self.detections_pedestrians_waiting = temp[0]  # [Detections, Detections, ...]

        temp = self.tracker["斑馬線"].getInside(
            detections,
            annotated_frame,
            byTime=0,
            class_name=["pedestrian"],
            labelFunc=lambda x: "Crossing" if x > 0 else "Nobody",
        )

        self.detections_pedestrians_crossing = temp[0]  # [Detections, Detections, ...]

        temp = self.tracker["斑馬線"].getInside(
            detections,
            annotated_frame,
            byTime=0,
            class_name=["vehicle"],
            labelFunc=lambda x: F"Vehicle:{x}",
            TextOffsetY=50,
        )

        self.detections_vehicle_entered = temp[0]  # [Detections, Detections, ...]

        return annotated_frame

    def getViolateVehicle(self, signal="行人紅燈"):
        if signal == "行人紅燈":
            if (
                len(self.detections_vehicle_entered) > 0
                and len(self.detections_pedestrians_crossing) > 0
            ):
                # 行人紅燈 但還有行人還在過馬路
                pass
        elif signal == "行人綠燈":
            if len(self.detections_vehicle_entered) > 0 and (
                len(self.detections_pedestrians_waiting) > 0
                or len(self.detections_pedestrians_crossing) > 0
            ):
                # 行人綠燈 但有車無停下等行人通過
                pass
        elif signal == "無號誌":
            if len(self.detections_vehicle_entered) > 0 and (
                len(self.detections_pedestrians_waiting) > 0
                or len(self.detections_pedestrians_crossing) > 0
            ):
                # 無號誌 但有車無停下等行人通過
                pass

        return self.detections_vehicle_entered


window = Window("Preview")
triangleA_annotator = sv.TriangleAnnotator()


def main():
    video_path = "./video/WIN_20231229_11_33_12_Pro.mp4"
    video_info = sv.VideoInfo.from_video_path(video_path=video_path)
    cap = cv2.VideoCapture(video_path)
    # show first frame
    first_frame = cap.read()[1]
    window.image = first_frame

    walkerGuard = WalkerGuard("./model/vir.pt", window, video_info)

    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
    while cap.isOpened():
        ret, frame = cap.read()
        if ret:
            detections, results = walkerGuard.detector.detect(frame, conf=0.2, verbose=False)
            frame = triangleA_annotator.annotate(frame, detections)
            # Detection every area's objects
            image = walkerGuard.update(frame, detections)
            # Violate the objects

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
