from collections import defaultdict

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
import time

############some variables initialization
shape = (400, 400, 3)  # y, x, RGB
origin_img = np.full(shape, 255).astype(np.uint8)
colors = sv.ColorPalette.default()
ax = []
ay = []
a_ped = []
a_car = []
# plt.ion()
fig = plt.figure()
frame_count = 0
# opencv
close_mask = 0
# Load the YOLOv8 model
model = YOLO('./model/vir.pt')
# print(model.fuse())

# Open the video file
video_path = "./video/japan.mp4"
video_info = sv.VideoInfo.from_video_path(video_path=video_path)
print(video_info)

class_list: dict = model.model.names  # 取得所有class的名稱
#: 'pedestrian', 1: 'people', 2: 'bicycle', 3: 'car', 4: 'van', 5: 'truck', 6: 'tricycle', 7: 'awning-tricycle', 8: 'bus', 9: 'motor'}


box_annotator = sv.BoundingBoxAnnotator(thickness=2, color=ColorPalette.default())
label_annotator = sv.LabelAnnotator()
trace_annotator = sv.TraceAnnotator()
tracker = sv.ByteTrack()

cap = cv2.VideoCapture(video_path)
detections_p_previous = None
# x811 y193
# x1223 y309
# 371左上y565右下x1066右下y712
LINE_START = sv.Point(371, 565)
LINE_END = sv.Point(1066, 712)
line_counter = sv.LineZone(start=LINE_START, end=LINE_END)
testline_annotator = sv.LineZoneAnnotator()
# poly


f = open('./polygon.txt', 'r')
p = f.read()
#############some flag variables
TRAFFIC_FLAG = False
IS_WAIT_FLAG = False
IS_RED = False
#############斑馬線區域
polygon_np = [
    # np.array([[149, 287], [124, 247], [411, 130], [479, 139]]),
    np.array([[816, 186], [1224, 298], [1064, 710], [644, 714], [388, 562]]),
]
zones = [
    sv.PolygonZone(polygon=polygon, frame_resolution_wh=video_info.resolution_wh)
    for polygon in polygon_np
]
zone_annotators = [
    sv.PolygonZoneAnnotator(
        zone=zone, color=colors.by_idx(index), thickness=2, text_thickness=5, text_scale=2
    )
    for index, zone in enumerate(zones)
]
zone_box_annotators = [
    sv.TriangleAnnotator(
        color=colors.by_idx(index),
    )
    for index in range(len(polygon_np))
]
zone_TRACK_annotators = [sv.ByteTrack() for index in range(len(polygon_np))]
zone_trac_annotators = [
    sv.TraceAnnotator(color=colors.by_idx(index)) for index in range(len(polygon_np))
]
#############行人等待區區域
Pedestrian_Poly = [
    np.array([[843, 194], [889, 172], [1198, 249], [1203, 283]]),
    np.array([[451, 711], [257, 706], [329, 605]]),
]
Ped_zones = [
    sv.PolygonZone(polygon=polygon, frame_resolution_wh=video_info.resolution_wh)
    for polygon in Pedestrian_Poly
]
Ped_zone_annotators = [
    sv.PolygonZoneAnnotator(
        zone=zone, color=colors.by_idx(index + 4), thickness=2, text_thickness=5, text_scale=2
    )
    for index, zone in enumerate(Ped_zones)
]


######################
####some function#####
def moving_avg(x, n):
    cumsum = np.cumsum(np.insert(x, 0, 0))
    return (cumsum[n:] - cumsum[:-n]) / float(n)


def traffic(val):
    global TRAFFIC_FLAG
    if val != 0:
        TRAFFIC_FLAG = 1
    else:
        TRAFFIC_FLAG = 0


def test(val):
    global close_mask
    if val != 0:
        close_mask = 1
    else:
        close_mask = 0
    print(close_mask)


def plot_update(temp_frame):  # 合併圖表跟影像 原理是抓圖一的寬高跟圖二寬高 然後高度取最大 寬度相加(所以是左右concat)
    image = cv2.imread('matplotlib_plot.png')
    h1, w1 = image.shape[:2]
    h2, w2 = temp_frame.shape[:2]

    # create empty matrix
    vis = np.zeros((max(h1, h2), w1 + w2, 3), np.uint8)

    # combine 2 images
    vis[:h1, :w1, :3] = image
    vis[:h2, w1 : w1 + w2, :3] = temp_frame
    return vis


def update_plot_statistics():
    # fig.clf()
    plt.legend(['InCrossRoad', 'pedestrian', 'car'])
    if len(ax) > 30:
        ax.pop(0)
        ay.pop(0)
        a_ped.pop(0)
        a_car.pop(0)
        tempx = np.array(ax)
        tempy = np.array(ay)
        tempp = np.array(a_ped)
        tempc = np.array(a_car)
        mod = make_interp_spline(tempx, tempy)
        mod2 = make_interp_spline(tempx, tempp)
        mod3 = make_interp_spline(tempx, tempc)
        x2 = np.linspace(tempx.min(), tempx.max(), 500)
        y2 = mod(x2)
        p2 = mod2(x2)
        c2 = mod3(x2)
        plt.plot(x2, y2)
        plt.plot(x2, p2)
        plt.plot(x2, c2)
    else:
        pass
        plt.plot(ax, ay)
        plt.plot(ax, a_ped)
        plt.plot(ax, a_car)
    plt.savefig('matplotlib_plot.png')
    plt.clf()


def Pedestrian_Update(detections, annotated_frame):
    Pedestrian_Area = []  # 如果有多個區域 其實就是把每個區域個別抓出來
    Total_in_area = 0
    for zone, zone_annotator in zip(Ped_zones, Ped_zone_annotators):
        Total_in_area += zone.current_count
        zone = zone.trigger(detections=detections)
        count_for_pedestrian = len([i for i in detections.class_id if i == 0])
        annotated_frame = zone_annotator.annotate(
            scene=frame, label="Waiting" if count_for_pedestrian > 0 else "No Wait"
        )
        detection_temp = detections[zone]  # 取出該區域的物件
        Pedestrian_Area.append(detection_temp)  # 把該區域的物件加入list
    detections = sv.Detections.merge(Pedestrian_Area)  # 最後把區域的物整合 (如果無號誌合併行人區斑馬線區)
    detections_InCrossRoad = sv.Detections.merge(Pedestrian_Area)
    return detections, Total_in_area, annotated_frame


def InCrossRoad_Update(detections, annotated_frame):
    InCrossRoad = []  # 如果有多個區域 其實就是把每個區域個別抓出來
    Total_in_area = 0
    for zone, zone_annotator, box_annotator, trackv, trac_annotator in zip(
        zones, zone_annotators, zone_box_annotators, zone_TRACK_annotators, zone_trac_annotators
    ):
        Total_in_area += zone.current_count
        zone = zone.trigger(detections=detections)
        detection_temp = detections[zone]  # 取出該區域的物件
        count_for_pedestrian = len([i for i in detection_temp.class_id if i == 0])
        annotated_frame = zone_annotator.annotate(scene=frame, label=F"{count_for_pedestrian}")

        annotated_frame = box_annotator.annotate(scene=annotated_frame, detections=detection_temp)
        detection_temp = trackv.update_with_detections(detection_temp)
        annotated_frame = trac_annotator.annotate(scene=annotated_frame, detections=detection_temp)
        InCrossRoad.append(detection_temp)  # 把該區域的物件加入list
    detections = sv.Detections.merge(InCrossRoad)  # 最後把區域的物整合 (如果無號誌合併行人區斑馬線區)
    return detections, Total_in_area, annotated_frame, InCrossRoad  # 回傳各區斑馬線的detection 因為要分開做追蹤


with sv.VideoSink(target_path='abc.mp4', video_info=video_info) as sink:
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        results = model.predict(frame, conf=0.5, verbose=False)[0]  # 可設定最小要幾趴 aka threshold

        # SET FLAG
        IS_WAIT_FLAG = False

        detections = sv.Detections.from_ultralytics(results)  # 取得偵測到的物件
        detections_origin = detections  # 原始的detections存一個備用
        pedestrian_count = 0
        car_count = 0

        for element in detections.class_id:
            if class_list[element] != 'pedestrian':
                car_count += 1
            else:
                pedestrian_count += 1
        # 這個code大概隔週看就忘嘞x

        if close_mask == 0:  # 更新斑馬線區
            detections_i, Total_in_area, annotated_frame, incrossRoad = InCrossRoad_Update(
                detections_origin, frame
            )
            # detections_i所有在斑馬線區的物件
            detections_p, Total_in_area2, annotated_frame = Pedestrian_Update(
                detections_origin, annotated_frame
            )
            # detections_p所有在行人區的物件
            detections = sv.Detections.merge([detections_i, detections_p])
            # 如果有號誌only斑馬線區

        detections = tracker.update_with_detections(detections)
        # annotated_frame = trace_annotator.annotate(#標記追蹤線
        #     scene=annotated_frame,
        #     detections=detections,
        # )
        # labels = [
        #     f"{results.names[class_id]}{confidence:0.2f}#{tracker_id}"
        #     for xyxy,mask,confidence,class_id,tracker_id
        #     in detections
        # ]

        # annotated_frame = label_annotator.annotate(#標記 名稱阿 id阿 在label裡自訂義
        #     scene=annotated_frame,
        #     detections=detections,
        #     labels=labels
        # )

        # 到時加if 行人數量>2且通過就警示
        detections_p = tracker.update_with_detections(detections_p)
        if frame_count != 0:
            for xyxy, mask, confidence, class_id, tracker_id in detections_p:
                last_tracker_id = detections_p_previous.tracker_id
                anchor_now_x = xyxy[2] - xyxy[0]
                anchor_now_y = xyxy[3] - xyxy[1]
                # print(anchor_now_x,anchor_now_y)
                if tracker_id in last_tracker_id:
                    index = list(last_tracker_id).index(tracker_id)
                    last_xyxy = detections_p_previous.xyxy[index]
                    anchor_pre_x = last_xyxy[2] - last_xyxy[0]
                    anchor_pre_y = last_xyxy[3] - last_xyxy[1]
                    distance = np.sqrt(
                        (anchor_pre_x - anchor_now_x) ** 2 + (anchor_pre_y - anchor_now_y) ** 2
                    )
                    if distance == 0:  # 迷失追蹤
                        continue
                    elif distance < 0.15:
                        pass
                        # print('有人停下來了')
                    elif distance < 0.05:
                        IS_WAIT_FLAG = True
                        print('有人超級趨近於0')
                    # print(distance)
                # print(last_tracker_id)
                last_xyxy = detections_p_previous.xyxy
        print("------------------")
        for incross in incrossRoad:  # 每區的斑馬線個別確認有沒有車 避免a沒人 b區有人 可是a區車通過被誤判
            classes_incross = [class_list[i] for i in incross.class_id]
            classes_inwait = [class_list[i] for i in detections_p.class_id]
            # print(
            #     "\n".join(
            #         [
            #             F"{name}:{len([x for x in classes_incross if x == name])}"
            #             for name in class_list.values()
            #             if len([x for x in classes_incross if x == name]) != 0
            #         ]
            #     )
            # )

        # for element in detections.class_id:
        #     if class_list[element]!='pedestrian' : #有'pedestrian' 跟 'people' 但people連機車上的人也會偵測 但行人不會被誤判
        #         if not TRAFFIC_FLAG and IS_WAIT_FLAG :
        #             print("有車違規-無號誌模式-有人在等待")
        # 號誌undo

        # 兩種情況 =>有號誌 if 綠燈=true 就只偵測斑馬線區 else都抓
        # 兩種情況 =>無號誌 if 都抓 and 看行人是否等待或是路過

        if close_mask == 1:
            detections = detections_origin
            annotated_frame = box_annotator.annotate(  # 標記bounding box
                scene=frame, detections=detections
            )

        # print(detections.tracker_id)

        detections_p_previous = tracker.update_with_detections(detections_p)

        frame_count += 1
        ay.append(Total_in_area)
        a_ped.append(pedestrian_count)
        a_car.append(car_count)
        ax.append(frame_count)

        # plt.pause(0.0100000001)
        # plt.ioff()
        # update_plot_statistics()

        # sink.write_frame(frame=annotated_frame)
        annotated_frame = plot_update(annotated_frame)
        # print(frame_count)
        starttime = time.time()
        plt.clf()
        cv2.imshow('frame', annotated_frame)
        cv2.imshow('GUI', origin_img)
        if frame_count == 1:
            cv2.createTrackbar('Right-0-CloseMask', 'GUI', 0, 1, test)
            cv2.createTrackbar('PLOT_DISABLE', 'GUI', 0, 1, test)
            cv2.createTrackbar('traffic_signal', 'GUI', 0, 1, test)
            cv2.createTrackbar('red_green', 'GUI', 0, 1, test)
        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            break
        elif key == ord("s"):
            cap.set(1, cap.get(1) + 30 * 5)
cap.release()
cv2.destroyAllWindows()
