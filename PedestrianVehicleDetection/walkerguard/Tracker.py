from collections import defaultdict
from pprint import pprint
from typing import Dict
import numpy as np
import supervision as sv
from supervision import Detections, BoxAnnotator
from supervision import ColorPalette
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import datetime


colors = sv.ColorPalette.default()


class Tracker:
    def __init__(
        self,
        poly: list[np.ndarray],
        class_dict: dict,
        video_info: sv.VideoInfo,
    ):
        self.class_dict: class_dict = class_dict
        self.poly = [
            sv.PolygonZone(polygon=polygon, frame_resolution_wh=video_info.resolution_wh)
            for polygon in poly
        ]
        self.zone_annotators = [
            sv.PolygonZoneAnnotator(
                zone=poly, color=colors.by_idx(i), thickness=2, text_thickness=1, text_scale=1
            )
            for i, poly in enumerate(self.poly)
        ]
        self.track_annotators = [sv.ByteTrack() for i in range(len(self.poly))]
        self.trackingStartTime = [
            defaultdict(lambda: datetime.datetime.now()) for i in range(len(self.poly))
        ]
        self.box_annotator = sv.BoundingBoxAnnotator(thickness=2, color=ColorPalette.default())
        self.label_annotator = sv.LabelAnnotator(text_position=sv.Position.TOP_LEFT)

    def getInside(
        self,
        detections: Detections,
        annotated_frame=None,
        byTime=0,
        class_name="",
        labelFunc: callable = None,
    ) -> tuple[Detections, list[int], np.ndarray]:
        output: list[Detections] = []
        output_count: list[int] = []
        print("detections : ")
        for zone_index, (zone, track_annotator, zone_annotator) in enumerate(
            zip(self.poly, self.track_annotators, self.zone_annotators)
        ):
            inside_tracker_id = []
            inside_detections = []
            zone = zone.trigger(detections=detections)
            detection_temp: Detections = detections[zone]  # 取出該區域的物件
            detection_temp = track_annotator.update_with_detections(detection_temp)

            for xyxy, mask, confidence, class_id, tracker_id in detection_temp:
                inside_tracker_id.append(tracker_id)
                print(inside_tracker_id)
                if (
                    self.trackingStartTime[zone_index][tracker_id]
                    + datetime.timedelta(seconds=byTime)
                    < datetime.datetime.now()
                ):
                    if class_name == "" or self.class_dict[class_id] == class_name:
                        inside_detections.append([xyxy, mask, confidence, class_id, tracker_id])
            ObjCount = len(inside_detections)
            output_count.append(ObjCount)
            try:
                # visualize the objects
                if annotated_frame is not None:
                    annotated_frame = zone_annotator.annotate(
                        scene=annotated_frame,
                        label=ObjCount if labelFunc is None else labelFunc(ObjCount),
                    )

                inside_detections = sv.Detections(
                    xyxy=np.array([x[0] for x in inside_detections]),
                    confidence=np.array([x[2] for x in inside_detections]),
                    class_id=np.array([x[3] for x in inside_detections]),
                    tracker_id=np.array([x[4] for x in inside_detections]),
                )

                if annotated_frame is not None:
                    annotated_frame = self.box_annotator.annotate(
                        scene=annotated_frame, detections=inside_detections
                    )
                    annotated_frame = self.label_annotator.annotate(
                        scene=annotated_frame,
                        detections=inside_detections,
                        labels=[
                            F"{self.class_dict[x]} # {tid}"
                            for x, tid in zip(
                                inside_detections.class_id, inside_detections.tracker_id
                            )
                        ],
                    )
                output.append(inside_detections)

            except Exception as e:
                output.append(None)

            for tracker_id in list(self.trackingStartTime[zone_index].keys()):
                if tracker_id not in inside_tracker_id:
                    self.trackingStartTime[zone_index].pop(tracker_id)

        return (output, output_count, annotated_frame)
