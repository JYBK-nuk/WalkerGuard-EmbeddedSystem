import os
print(os.environ.get('PATH'))

import cv2
from ultralytics import YOLO
import plate

# Load the YOLOv8 model
model = YOLO('./best.pt')

# Open the video file
video_path = "IMG_5306.MOV"
cap = cv2.VideoCapture(video_path)

# Loop through the video frames
while cap.isOpened():
    # Read a frame from the video
    success, frame = cap.read()

    if success:
        # Run YOLOv8 inference on the frame
        results = model(frame, verbose=False)

        # Visualize the results on the frame
        annotated_frame = results[0].plot()

        # Get the boxes and track IDs
        boxes = results[0].boxes.xyxy.cpu()
        classes = results[0].boxes.cls.int().cpu().tolist()

        annotated_frame = cv2.resize(
            annotated_frame, (1600, int(1600 * annotated_frame.shape[0] / annotated_frame.shape[1]))
        )

        # Display the annotated frame
        cv2.imshow("YOLOv8 Inference", annotated_frame)

        # 抓車牌
        Plates = []
        for box, class_id in zip(boxes, classes):
            x, y, x2, y2 = box
            if results[0].names[class_id] == 'Plate':
                Plates.append(frame[int(y) : int(y2), int(x) : int(x2)])
        if len(Plates) > 0:
            # combine images into a single image
            for i in range(len(Plates)):
                Plates[i] = cv2.resize(Plates[i], (200, 200))

        plate.get_all_plates(Plates)




        # Break the loop if 'q' is pressed
        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            break
        elif key == ord("s"):
            plate.get_all_plates(Plates)
        elif key == ord("c"): # skip 5 sec
            cap.set(cv2.CAP_PROP_POS_FRAMES, cap.get(cv2.CAP_PROP_POS_FRAMES) + 150)

    else:
        # Break the loop if the end of the video is reached
        break

# Release the video capture object and close the display window
cap.release()
cv2.destroyAllWindows()
