from collections import defaultdict

import cv2
import numpy as np

from ultralytics import YOLO
import supervision as sv
from supervision import Detections,BoxAnnotator
from supervision import ColorPalette
from supervision import Color
from supervision import Point
# Load the YOLOv8 model
import tkinter as tk
from PIL import Image, ImageTk
from matplotlib import pyplot as plt
from matplotlib.figure import Figure
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

ax = []                    
ay = []                   
plt.ion() 
frame_count=0
window = tk.Tk()

frame = tk.Frame(window)                  # 加入 Frame 框架
frame.pack()
a = tk.StringVar() 
def show(e):
    if scale_h.get() ==0:
        a.set('無標誌')
    if scale_h.get() ==1:
        if scale_b.get() == 0:
            a.set('紅燈')
            scale_b.configure(background='red')
        else:
            a.set('綠燈')
            scale_b.configure(background='green')
label = tk.Label(window, textvariable=a)
label.pack()
scale_h = tk.Scale(frame, from_=0, to=1, orient='horizontal',command=show)  # 改變時執行 show
scale_h.pack(side=tk.LEFT)
scale_b = tk.Scale(frame, from_=0, to=1, orient='horizontal',command=show)  # 改變時執行 show
scale_b.pack(side=tk.LEFT)



model = YOLO('./PedestrianVehicleDetection/model/vir.pt')
#print(model.fuse())

# Open the video file
video_path = "./video/japan.mp4"
video_info = sv.VideoInfo.from_video_path(video_path=video_path)
print(video_info)

class_list = model.model.names#取得所有class的名稱
#: 'pedestrian', 1: 'people', 2: 'bicycle', 3: 'car', 4: 'van', 5: 'truck', 6: 'tricycle', 7: 'awning-tricycle', 8: 'bus', 9: 'motor'}


box_annotator = sv.BoundingBoxAnnotator(thickness= 2, color=ColorPalette.default())
label_annotator = sv.LabelAnnotator()
trace_annotator = sv.TraceAnnotator()
tracker=sv.ByteTrack()

cap = cv2.VideoCapture(video_path)

#x811 y193
#x1223 y309
#371左上y565右下x1066右下y712
LINE_START = sv.Point(371, 565)
LINE_END = sv.Point(1066, 712)
line_counter=sv.LineZone(start=LINE_START, end=LINE_END)
testline_annotator = sv.LineZoneAnnotator()
#poly


polygon=sv.PolygonZone(polygon=np.array([
        [816, 186],[1224, 298],[1064, 710],[644, 714],[388, 562]
    ]),frame_resolution_wh=video_info.resolution_wh)

polygon_annotator = sv.PolygonZoneAnnotator(zone=polygon,
        color=Color(0,255,0),
        thickness=4,
        text_thickness=8,
        text_scale=4)

def update_image():
    global frame_count
    with sv.VideoSink(target_path='abc.mp4', video_info=video_info) as sink:
        ret, frame = cap.read()
        # if frame is read correctly ret is True
        results=model.predict(frame,conf=0.5)[0]#可設定最小要幾趴 aka threshold
        
        detections = sv.Detections.from_ultralytics(results)#取得偵測到的物件
        
        #這個code大概隔週看就忘嘞x
        frame_count+=1
        InCrossRoad=[]#如果有多個區域 其實就是把每個區域個別抓出來
        InCrossRoadMask=polygon.trigger(detections=detections)#他把所有在區域內的物件都抓出來
        InCrossRoadin=detections[InCrossRoadMask]#把本來的偵測到的(區域內外)的只抓出在區域內的
        InCrossRoad.append(InCrossRoadin)
        detections=sv.Detections.merge(InCrossRoad)#最後把區域的物整合
        temp=0
        for element in detections.class_id:
            temp+=1
            if class_list[element]!='pedestrian': #有'pedestrian' 跟 'people' 但people連機車上的人也會偵測 但行人不會被誤判 
                print("有車")

        annotated_frame = polygon_annotator.annotate(scene=frame)#劃出斑馬線區域
        
        annotated_frame = box_annotator.annotate(#標記bounding box
            scene=annotated_frame,
            detections=detections
        )
        
        detections = tracker.update_with_detections(detections)
        annotated_frame = trace_annotator.annotate(#標記追蹤線
            scene=annotated_frame,
            detections=detections
        )
        
        labels = [
            f"{results.names[class_id]}{confidence:0.2f}#{tracker_id}"
            for xyxy,mask,confidence,class_id,tracker_id
            in detections
        ]
        
        annotated_frame = label_annotator.annotate(#標記 名稱阿 id阿 在label裡自訂義
            scene=annotated_frame,
            detections=detections,
            labels=labels
        )
        
        #annotated_frame = sv.draw_line(scene=annotated_frame, start=LINE_START, end=LINE_END,color=Color(255,0,0), thickness=4)
        #annotated_frame=testline_annotator.annotate(annotated_frame, line_counter)
        #line_counter.trigger(detections=detections)
        
        sink.write_frame(frame=annotated_frame)
        annotated_frame = cv2.cvtColor(annotated_frame, cv2.COLOR_BGR2RGB)
        img = Image.fromarray(annotated_frame)
        # 将PIL图像转换为Tkinter图像
        img_tk = ImageTk.PhotoImage(image=img)
        ay.append(polygon.current_count)           
        ax.append(frame_count)        
        if len(ax) >30:
            ax.pop(0)
            ay.pop(0)
        plt.clf()              
        plt.plot(ax,ay)        
        plt.pause(0.1)         
        plt.ioff()
        label.configure(image=img_tk)
        label.image = img_tk
        label.after(10, update_image)  
    

            
        
label = tk.Label(window)
label.pack()

update_image()
frame_count+=1
window.mainloop()
cap.release()
cv2.destroyAllWindows()
    
    