from collections import defaultdict
import os
import cv2
import numpy as np

from ultralytics import YOLO

models = []
for file in os.listdir("model"):
    if file.endswith(".pt"):
        models.append(file)
print("Choose model:")
print("\n".join([f"{i+1}: {model}" for i, model in enumerate(models)]))

# Load the YOLOv8 model
model = YOLO(F"model/{models[int(input())-1]}")

# Open the video file
video_path = "https://cctv4.kctmc.nat.gov.tw/47009f84" 
video_path = "https://cctv1.kctmc.nat.gov.tw/4c153727"
cap = cv2.VideoCapture(video_path)

# Store the track history
# defaultdict
# 就算是不存在的 track_id 也會有一個空的 list
# 就不用檢查 track_id in track_history 在 append 了
track_history = defaultdict(lambda: {"class": None, "track": []})

# Loop through the video frames
while cap.isOpened():
    # Read a frame from the video
    success, frame = cap.read()

    if success:
        # Run YOLOv8 tracking on the frame, persisting tracks between frames
        results = model.track(frame, persist=True, verbose=False)
        if results[0].boxes.id is None:
            cv2.imshow("YOLOv8 Tracking", frame)
        else:
            # Get the boxes and track IDs
            boxes = results[0].boxes.xywh.cpu()
            track_ids = results[0].boxes.id.int().cpu().tolist()
            classes = results[0].boxes.cls.int().cpu().tolist()


            # Visualize the results on the frame
            annotated_frame = results[0].plot()

            # Plot the tracks
            for box, track_id, class_id in zip(boxes, track_ids, classes):
                x, y, w, h = box
                track = track_history[track_id]
                track["class"] = results[0].names[class_id]
                track["track"].append((float(x), float(y)))  # x, y center point
                if len(track) > 30:  # retain 90 tracks for 90 frames
                    track.pop(0)

                # Draw the tracking lines
                points = np.hstack(track["track"]).astype(np.int32).reshape((-1, 1, 2))
                cv2.polylines(
                    annotated_frame, [points], isClosed=False, color=(230, 230, 230), thickness=10
                )

            # Display the annotated frame
            cv2.imshow("YOLOv8 Tracking", annotated_frame)

        # Break the loop if 'q' is pressed
        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            break
        elif key == ord("s"):
            print("\n".join([f"{str(k)}: {v['class']}" for k, v in dict(track_history).items()]))
    else:
        # Break the loop if the end of the video is reached
        break

# Release the video capture object and close the display window
cap.release()
cv2.destroyAllWindows()
