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
from ui import UI
import time
import threading

from walkerguard import Detector, Tracker, Window


def getAllZoneHasDetections(detections: list[Detections]):
    for detection in detections:
        if detection is not None:
            return True
    return False


class WalkerGuard:
    detector: Detector
    corner_annotator = sv.BoxCornerAnnotator(corner_length=10, thickness=2)
    label_annotator = sv.LabelAnnotator(
        text_position=Position.TOP_LEFT,
        text_thickness=1,
        text_scale=0.35,
    )
    tracker = {}
    violate_history = {}

    def __init__(self, model_path: str, window: Window, video_info):
        self.detector = Detector(model_path)
        self.video_info = video_info
        self.trace_annotator = sv.TraceAnnotator(
            trace_length=video_info.fps * 10, position=Position.BOTTOM_CENTER
        )
        print(self.detector.class_dict)
        self.tracker = {
            "行人等待區": Tracker(
                poly=[
                    np.array(
                        # [(322, 324), (238, 357), (351, 395), (392, 335)]
                        window.getClickPoints(4),
                    ),
                    np.array(
                        # [(552, 177), (526, 191), (597, 201), (601, 180)]
                        window.getClickPoints(4),
                    ),
                ],
                class_dict=self.detector.class_dict,
                video_info=video_info,
                color=[Color(255, 255, 255), Color(255, 255, 255)],
            ),
            "斑馬線": Tracker(
                poly=[
                    np.array(
                        # [(527, 195), (323, 319), (408, 331), (587, 199)]
                        window.getClickPoints(4),
                    ),
                ],
                class_dict=self.detector.class_dict,
                video_info=video_info,
                color=[Color(200, 255, 255)],
            ),
        }

        self.detections_pedestrians_waiting = []
        self.detections_pedestrians_crossing = []
        self.detections_vehicle_entered = []

    def update(self, frame: np.ndarray, detections: Detections):
        self.corner_annotator.annotate(frame, detections)
        # self.label_annotator.annotate(
        #     scene=frame,
        #     detections=detections,
        #     labels=[
        #         F"{self.detector.class_dict[x]} #{tid}"
        #         for x, tid in zip(detections.class_id, detections.tracker_id)
        #     ],
        # )
        self.trace_annotator.annotate(scene=frame, detections=detections)

        temp = self.tracker["行人等待區"].getInside(
            detections,
            frame,
            byTime=1,
            class_name=["people", "pedestrian"],
            labelFunc=lambda x: "Waiting" if x > 0 else "Nobody",
            TextOffsetY=50,
        )
        self.detections_pedestrians_waiting = temp[0]  # [Detections, Detections, ...]

        temp = self.tracker["斑馬線"].getInside(
            detections,
            frame,
            byTime=0,
            class_name=["pedestrian"],
            labelFunc=lambda x: "Crossing" if x > 0 else "Nobody",
            TextOffsetY=50,
        )

        self.detections_pedestrians_crossing = temp[0]  # [Detections, Detections, ...]

        temp = self.tracker["斑馬線"].getInside(
            detections,
            frame,
            byTime=0,
            class_name=["vehicle"],
            labelFunc=lambda x: F"Vehicle:{x}",
            TextOffsetY=0,
        )

        self.detections_vehicle_entered = temp[0]  # [Detections, Detections, ...]

        return frame

    def __UpdateViolateHistory(self, detectionsList: list[Detections]):
        for detections in detectionsList:
            for t_id, class_id in zip(detections.tracker_id, detections.class_id):
                objectTraces = self.trace_annotator.trace.get(t_id)
                if objectTraces is not None:
                    if t_id not in self.violate_history:
                        window.showText(
                            [F"違規車輛: #{t_id} {self.detector.class_dict[class_id]}"],
                            (255, 0, 255),
                            3,
                        )
                        print(objectTraces)
                        self.violate_history[t_id] = objectTraces

    def getViolateVehicle(self, signal="行人紅燈"):
        if signal == "行人紅燈":
            if getAllZoneHasDetections(self.detections_vehicle_entered) and getAllZoneHasDetections(
                self.detections_pedestrians_crossing
            ):
                # 行人紅燈 但還有行人還在過馬路
                self.__UpdateViolateHistory(self.detections_vehicle_entered)

                pass
        elif signal == "行人綠燈":
            if getAllZoneHasDetections(self.detections_vehicle_entered) and (
                getAllZoneHasDetections(self.detections_pedestrians_waiting)
                or getAllZoneHasDetections(self.detections_pedestrians_crossing)
            ):
                # 行人綠燈 但有車無停下等行人通過
                self.__UpdateViolateHistory(self.detections_vehicle_entered)
                pass
        elif signal == "無號誌":
            if getAllZoneHasDetections(self.detections_vehicle_entered) and (
                getAllZoneHasDetections(self.detections_pedestrians_waiting)
                or getAllZoneHasDetections(self.detections_pedestrians_crossing)
            ):
                # 無號誌 但有車無停下等行人通過
                self.__UpdateViolateHistory(self.detections_vehicle_entered)
                pass

        return self.violate_history


window = Window("Preview")
print("Loading video...")
# video_path = 1
video_path = "./video/WIN_20231229_11_33_12_Pro.mp4"
if isinstance(video_path, int):
    cap = cv2.VideoCapture(video_path, cv2.CAP_DSHOW)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
    video_info = sv.VideoInfo(
        fps=cap.get(cv2.CAP_PROP_FPS),
        width=int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
        height=int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
    )
else:
    cap = cv2.VideoCapture(video_path)
    video_info = sv.VideoInfo.from_video_path(video_path)
print(video_info)


def main():
    import requests

    # show first frame
    while cap.isOpened():
        ret, frame = cap.read()
        if ret:
            window.image = frame
            break
    # post image to /initPlateCam/top
    # @app.post("/initPlateCam/{cam}")
    # async def process_image(image: UploadFile = File(...)
    try:
        file = cv2.imencode(".jpg", frame)[1].tobytes()
        requests.post("http://127.0.0.1/initPlateCam/top", files={"image": file})
    except:
        pass
    walkerGuard = WalkerGuard("./model/vir.pt", window, video_info)

    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
    while cap.isOpened():
        ret, frame = cap.read()
        if ret:
            detections, results = walkerGuard.detector.detect(frame, conf=0.1, verbose=False)

            ## Detection every area's objects and visualize
            annotated_frame = walkerGuard.update(frame.copy(), detections)

            ## get violate vehicles
            violate_vehicle = walkerGuard.getViolateVehicle(window.signal)

            ## refresh window
            if not isinstance(video_path, int):
                current_time = cap.get(cv2.CAP_PROP_POS_MSEC) / 1000
                annotated_frame = cv2.putText(
                    annotated_frame,
                    F"Time: {current_time:.2f}",
                    (10, video_info.resolution_wh[1] - 100),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1,
                    (255, 255, 255),
                    2,
                    cv2.LINE_AA,
                )
            window.update(annotated_frame)
            if window.key == ord('s'):
                # skip 5 seconds
                cap.set(cv2.CAP_PROP_POS_FRAMES, cap.get(cv2.CAP_PROP_POS_FRAMES) + 5 * 30)
            if window.key == ord('1'):
                jumpTime = 310
                cap.set(cv2.CAP_PROP_POS_FRAMES, jumpTime * 30)
            if window.key == ord('2'):
                pass
            if window.key == ord('i'):
                try:
                    file = cv2.imencode(".jpg", frame)[1].tobytes()
                    requests.post("http://127.0.0.1/initPlateCam/top", files={"image": file})
                except:
                    pass


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
