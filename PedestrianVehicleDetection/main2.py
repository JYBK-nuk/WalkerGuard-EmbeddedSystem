from collections import defaultdict

import cv2
import numpy as np
import matplotlib as mpl
mpl.use("tkagg")
from ultralytics import YOLO
import supervision as sv
from supervision import Detections,BoxAnnotator
from supervision import ColorPalette
from supervision import Color
from supervision import Point
from matplotlib import pyplot as plt
from matplotlib.figure import Figure
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from scipy.interpolate import make_interp_spline


############some variables initialization
shape = (400, 400, 3) # y, x, RGB
origin_img = np.full(shape, 255).astype(np.uint8)

ax = []      
ay = []
a_ped=[]
a_car=[]
#plt.ion() 
fig = plt.figure()
frame_count=0
#opencv
close_mask=0
TRAFFIC_FLAG=False
# Load the YOLOv8 model
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


f = open('./PedestrianVehicleDetection/polygon.txt', 'r')
p=f.read()
print(p)
polygon=sv.PolygonZone(polygon=np.array([
        [816, 186],[1224, 298],[1064, 710],[644, 714],[388, 562]
    ]),frame_resolution_wh=video_info.resolution_wh)
wait_area=sv.PolygonZone(polygon=np.array([
        [ 814,  194],[ 848,  165],[1205,  259],[1203,  303]
    ]),frame_resolution_wh=video_info.resolution_wh)



polygon_annotator = sv.PolygonZoneAnnotator(zone=polygon,
        color=Color(0,255,0),
        thickness=4,
        text_thickness=8,
        text_scale=4)
wait_area_polygon_annotator = sv.PolygonZoneAnnotator(zone=wait_area,
        color=Color(0,0,255),
        thickness=4,
        text_thickness=8,
        text_scale=4)
####some function
def moving_avg(x, n):
    cumsum = np.cumsum(np.insert(x, 0, 0)) 
    return (cumsum[n:] - cumsum[:-n]) / float(n)

def traffic(val):
    global TRAFFIC_FLAG
    if val!=0:
        TRAFFIC_FLAG=1
    else:
        TRAFFIC_FLAG=0
def test(val):
    global close_mask
    if val!=0:
        close_mask=1
    else:
        close_mask=0
    print(close_mask)
def plot_update(temp_frame):#合併圖表跟影像 原理是抓圖一的寬高跟圖二寬高 然後高度取最大 寬度相加(所以是左右concat)
    image = cv2.imread('matplotlib_plot.png')
    h1, w1 = image.shape[:2]
    h2, w2 = temp_frame.shape[:2]

    #create empty matrix
    vis = np.zeros((max(h1, h2), w1+w2,3), np.uint8)

    #combine 2 images
    vis[:h1, :w1,:3] = image
    vis[:h2, w1:w1+w2,:3] = temp_frame
    return vis
def update_plot_statistics():
    plt.clf()
    if len(ax) >30:
        ax.pop(0)
        ay.pop(0)
        a_ped.pop(0)
        a_car.pop(0)
        tempx=np.array(ax)
        tempy=np.array(ay)
        tempp=np.array(a_ped)
        tempc=np.array(a_car)
        mod=make_interp_spline(tempx,tempy)
        mod2=make_interp_spline(tempx,tempp)
        mod3=make_interp_spline(tempx,tempc)
        x2=np.linspace(tempx.min(), tempx.max(), 500)
        y2=mod(x2)
        p2=mod2(x2)
        c2=mod3(x2)
        plt.plot(x2,y2) 
        plt.plot(x2,p2) 
        plt.plot(x2,c2) 
        plt.legend(['inArea','pedestrian','car'])
    else:              
        plt.plot(ax,ay)
        plt.plot(ax,a_ped)
        plt.plot(ax,a_car)        
    plt.savefig('matplotlib_plot.png')


with sv.VideoSink(target_path='abc.mp4', video_info=video_info) as sink:
    while cap.isOpened():
        ret, frame = cap.read()
        # if frame is read correctly ret is True
        results=model.predict(frame,conf=0.5)[0]#可設定最小要幾趴 aka threshold
        
        detections = sv.Detections.from_ultralytics(results)#取得偵測到的物件
        pedestrian_count=0
        car_count=0
        for element in detections.class_id:
            if class_list[element]!='pedestrian': 
                car_count+=1
            else:
                pedestrian_count+=1
        #這個code大概隔週看就忘嘞x
        if close_mask==0:
            InCrossRoad=[]#如果有多個區域 其實就是把每個區域個別抓出來
            InCrossRoadMask=polygon.trigger(detections=detections)#他把所有在區域內的物件都抓出來
            wait_area_triggle=wait_area.trigger(detections=detections)#把在等待區的物件都抓出來
            InCrossRoadin=detections[InCrossRoadMask]#把本來的偵測到的(區域內外)的只抓出在區域內的
            InCrossRoad.append(InCrossRoadin)
            detections=sv.Detections.merge(InCrossRoad)#最後把區域的物整合 (如果無號誌合併行人區斑馬線區)
            #如果有號誌only斑馬線區
            
        #到時加if 行人數量>2且通過就警示
        for element in detections.class_id:
            if class_list[element]!='pedestrian': #有'pedestrian' 跟 'people' 但people連機車上的人也會偵測 但行人不會被誤判 
                print("有車")
        #兩種情況 =>有號誌 if 綠燈=true 就只偵測斑馬線區 else都抓
        #兩種情況 =>無號誌 if 都抓 and 看行人是否等待或是路過

        annotated_frame = polygon_annotator.annotate(scene=frame)#劃出斑馬線區域
        annotated_frame=wait_area_polygon_annotator.annotate(scene=annotated_frame)#劃出等待區域
        
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
        
        
        
        
        frame_count+=1
        ay.append(polygon.current_count)    
        a_ped.append(pedestrian_count)
        a_car.append(car_count)
        ax.append(frame_count)          
        
        #plt.pause(0.0100000001)        
        #plt.ioff()
        update_plot_statistics()
        
        sink.write_frame(frame=annotated_frame)
        annotated_frame=plot_update(annotated_frame)
        
        cv2.imshow('frame', annotated_frame)
        cv2.imshow('GUI', origin_img)
        if frame_count==1:
            cv2.createTrackbar('Right-0-CloseMask', 'GUI', 0, 1,test)
            cv2.createTrackbar('traffic_signal', 'GUI', 0, 1,test)
            cv2.createTrackbar('red_green', 'GUI', 0, 1,test)
        if cv2.waitKey(1) == ord('q'):
            break
cap.release()
cv2.destroyAllWindows()
    
    


