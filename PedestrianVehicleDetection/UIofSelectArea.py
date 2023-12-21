import cv2
import numpy as np
import random
img = cv2.VideoCapture('./video/japan.mp4')
img = img.read()[1]
#cv2.imwrite('output.png', img)

dots = []
polygon=[]
colorrr=(0,0,0)
# 滑鼠事件發生時要執行的函式
def show_xy(event,x,y,flags,param):
    global dot1, dot2, img         # 在函式內使用全域變數
        
    if event == 1:
        polygon.append([x, y])                          # 記錄座標
        cv2.circle(img, (x, y), 10, (random.randint(0,255),random.randint(0,255),255), -1)   # 在點擊的位置，繪製圓形
        num = len(polygon)                              # 目前有幾個座標
        if num > 1:                                  # 如果有兩個點以上
            x1 = polygon[num-2][0]
            y1 = polygon[num-2][1]
            x2 = polygon[num-1][0]
            y2 = polygon[num-1][1]
            cv2.line(img,(x1,y1),(x2,y2),colorrr,2)  # 取得最後的兩個座標，繪製直線
        cv2.imshow('chooseArea', img)
        print(polygon)

cv2.imshow('chooseArea', img)

first= True
while True:
    keycode = cv2.waitKey(0)  
    if keycode == 112:#press p開始  每次按p都會開始一個新的區域
        colorrr=(random.randint(0,255),random.randint(0,255),random.randint(0,255))
        cv2.setMouseCallback('chooseArea', show_xy)
        #print(np.array(polygon))
        if not first:
            dots.append(np.array(polygon))
            cv2.polylines(img, pts=[np.array(polygon)], isClosed=True, color=(0, 0, 255), thickness=3)
            cv2.fillPoly(img, [np.array(polygon)], 255)
            print(dots)
            polygon=[]
        else:
            first=False
    if keycode == 113:
        break
        

print(dots)
with open('./PedestrianVehicleDetection/polygon.txt', 'w') as f:
    f.write(str(dots))


cv2.destroyAllWindows()