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

class Detector:
    def __init__(self, model_path: str):
        self.model_path = model_path
        self.model = YOLO(self.model_path)
        self.class_dict: dict = self.model.model.names
        for key, value in self.class_dict.items():
            if value in ["car", "van", "bus", "truck", "motor"]:
                self.class_dict[key] = "vehicle"

        self.results = None

    def detect(self, image: np.ndarray, conf: float = 0.4, verbose: bool = False):
        results = self.model.predict(image, conf=conf, verbose=verbose)[0]
        detections = sv.Detections.from_ultralytics(results)
        return detections, results
