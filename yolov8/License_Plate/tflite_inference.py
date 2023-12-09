import cv2
import time
import numpy as np
import tensorflow as tf
model_path='./runs/detect/train/weights/best_saved_model/best_int8.tflite'
interpreter = tf.lite.Interpreter(model_path=model_path)
interpreter.allocate_tensors()

input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

# Obtain the height and width of the corresponding image from the input tensor
image_height = input_details[0]['shape'][1] # 640
image_width = input_details[0]['shape'][2] # 640

def show_detect(img , preds , iou_threshold , conf_threshold, class_label , color_palette):
    boxes = []
    scores = []
    class_ids = []
    
    # print()
    max_conf = np.max(preds[0,4:,:] , axis=0)
    idx_list = np.where(max_conf > conf_threshold)[0]
    
    # for pred_idx in range(preds.shape[2]):
    for pred_idx in idx_list:

        pred = preds[0,:,pred_idx]
        conf = pred[4:]
        
        
        box = [pred[0] - 0.5*pred[2], pred[1] - 0.5*pred[3] , pred[0] + 0.5*pred[2] , pred[1] + 0.5*pred[3]]
        boxes.append(box)

        label = np.argmax(conf)
        
        scores.append(max_conf[pred_idx])
        class_ids.append(label)

    boxes = np.array(boxes)
    result_boxes = non_maximum_suppression_fast(boxes, overlapThresh=iou_threshold)
    

    for i in range(len(result_boxes)):
        index = result_boxes[i]
        box = boxes[index]
        
        draw_detect(img, round(box[0]), round(box[1]),round(box[2]), round(box[3]),
            scores[index] , class_ids[index] , class_label , color_palette)
    
    return
def non_maximum_suppression_fast(boxes, overlapThresh=0.3):
    
    # If there is no bounding box, then return an empty list
    if len(boxes) == 0:
        return []
        
    # Initialize the list of picked indexes
    pick = []
    
    # Coordinates of bounding boxes
    x1 = boxes[:,0].astype("float")
    y1 = boxes[:,1].astype("float")
    x2 = boxes[:,2].astype("float")
    y2 = boxes[:,3].astype("float")
    
    # Calculate the area of bounding boxes
    bound_area = (x2-x1+1) * (y2-y1+1)
    # Sort the bounding boxes by the bottom-right y-coordinate of the bounding box
    sort_index = np.argsort(y2)
    
    
    # Looping until nothing left in sort_index
    while sort_index.shape[0] > 0:
        # Get the last index of sort_index
        # i.e. the index of bounding box having the biggest y2
        last = sort_index.shape[0]-1
        i = sort_index[last]
        
        # Add the index to the pick list
        pick.append(i)
        
        # Compared to every bounding box in one sitting
        xx1 = np.maximum(x1[i], x1[sort_index[:last]])
        yy1 = np.maximum(y1[i], y1[sort_index[:last]])
        xx2 = np.minimum(x2[i], x2[sort_index[:last]])
        yy2 = np.minimum(y2[i], y2[sort_index[:last]])        

        # Calculate the width and height of the bounding box
        w = np.maximum(0, xx2 - xx1 + 1)
        h = np.maximum(0, yy2 - yy1 + 1)
        # Compute the ratio of overlapping
        overlap = (w*h) / bound_area[sort_index[:last]]
        
        # Delete the bounding box with the ratio bigger than overlapThresh
        sort_index = np.delete(sort_index, 
                               np.concatenate(([last], np.where(overlap > overlapThresh)[0])))
        
    # return only the bounding boxes in pick list        
    # return boxes[pick]
    return pick


def draw_detect(img , x1 , y1 , x2 , y2 , conf , class_id , label , color_palette):
    # label = f'{CLASSES[class_id]} ({confidence:.2f})'
    color = color_palette[class_id]
    
    # print(x1 , y1 , x2 , y2 , conf , class_id)
    cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)

    cv2.putText(img, f"{label[class_id]} {conf:0.3}", (x1 - 10, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)



cap = cv2.VideoCapture(0)
while cap.isOpened():
    ret, frame = cap.read()
    image_resized = cv2.resize(frame, (image_height, image_width))

    image_np = np.array(image_resized) #
    image_np = np.true_divide(image_np, 255, dtype=np.float32) 
    image_np = image_np[np.newaxis, :]

    # inference
    interpreter.set_tensor(input_details[0]['index'], image_np)

    start = time.time()
    interpreter.invoke()

    # Obtaining output results
    output = interpreter.get_tensor(output_details[0]['index'])
    print(output.shape)
    
    show_detect(image_resized , output , 0.3 , 0.3 , ['licence'] , [(0,255,0)])
    end = time.time()
    fps = 1 / (end - start)
    cv2.putText(image_resized, str(fps), (0, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
    cv2.imshow('result', image_resized)
    if cv2.waitKey(1) == ord('q'):
        break