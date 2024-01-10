import base64
from pprint import pprint
import time
from fastapi import FastAPI, File, Request, UploadFile, WebSocket, WebSocketDisconnect
from fastapi.concurrency import asynccontextmanager
from fastapi.responses import FileResponse, HTMLResponse, StreamingResponse
import uvicorn
import cv2
import numpy as np
from typing import List
import pydantic
from PIL import Image
import asyncio
import json
import pickle

from plate import get_all_plates

app = FastAPI()

connected_clients = set()
isShutdown = False


class plate(pydantic.BaseModel):
    time: float
    plate: list[str]
    position: list[list[int]]


class plateNumpy:
    time: float
    plate: list[np.ndarray]
    position: list[list[int]]

    def __init__(self, time, plate, position):
        self.time = time
        self.plate = plate
        self.position = position


class Cameras:
    initPoints = {}
    PerspectiveTransforms = {}
    targetPolygon = [400, 300]
    images: dict[np.ndarray] = {}

    def __init__(self):
        # try to load from json
        try:
            # load with pickle
            with open("cameras.pickle", "rb") as f:
                cameras = pickle.load(f)
                self.initPoints = cameras.initPoints
                self.PerspectiveTransforms = cameras.PerspectiveTransforms
                self.targetPolygon = cameras.targetPolygon
                self.images = cameras.images

        except Exception as e:
            print(e)
            pass
        print("initPoints....")
        self.getAllMatrixAndTransformedImage()
        print(self.initPoints)
        print(self.PerspectiveTransforms)

    def save(self):
        # save with pickle
        with open("cameras.pickle", "wb") as f:
            pickle.dump(self, f)

    def add(self, cam: str, points: list[list[int]], image: np.ndarray):
        self.initPoints[cam] = points
        print(F"initPoints: {self.initPoints}")
        self.images[cam] = image
        self.save()

    def getAllMatrixAndTransformedImage(
        self,
    ):
        for key, value in self.initPoints.items():
            pts1 = np.float32(value)
            # 左上、左下、右下、右上 (左上為0,0)
            pts2 = np.float32(
                [
                    [0, 0],
                    [0, self.targetPolygon[1]],
                    [self.targetPolygon[0], self.targetPolygon[1]],
                    [self.targetPolygon[0], 0],
                ]
            )
            # 計算轉換矩陣
            M = cv2.getPerspectiveTransform(pts1, pts2)
            self.PerspectiveTransforms[key] = M

    def Transform(self, from_: str, to_: str, x: int, y: int):
        print(self.initPoints)
        print(self.PerspectiveTransforms)
        print(from_, to_)
        print(x, y)

        fromM = self.PerspectiveTransforms[from_]
        toM = self.PerspectiveTransforms[to_]
        transformed_x, transformed_y = cv2.perspectiveTransform(
            np.array([[[x, y]]], dtype=np.float32), fromM
        )[0][0]
        invToM = np.linalg.inv(toM)
        x, y = cv2.perspectiveTransform(
            np.array([[[transformed_x, transformed_y]]], dtype=np.float32), invToM
        )[0][0]
        return int(x), int(y)


cameras = Cameras()


plates: List[plateNumpy] = []


@app.on_event("shutdown")
def shutdown_event():
    isShutdown = True


@app.post("/violate")
async def violate(request: Request):
    data = await request.json()
    x = data["x"]
    y = data["y"]
    print(x, y)
    x, y = cameras.Transform("top", "plate", x, y)
    print("Transform")
    print(x, y)
    tragetImage = cameras.images["plate"].copy()
    # draw circle
    cv2.circle(tragetImage, (x, y), 5, (0, 0, 255), -1)
    # save image
    cv2.imshow("violate", tragetImage)
    cv2.waitKey(0)
    cv2.destroyAllWindows() 
    # return image
    return {"message": "violate"}


@app.get("/", response_class=HTMLResponse)
async def root():
    html_content = """
    <!DOCTYPE html>
    <html>
    <head>
        <title>違規車輛</title>
        <script>
            function updateImage() {
                var img = document.getElementById('image');
                fetch('/update');
            }
        </script>
    </head>
    <body>
        <div style="display: flex; flex-direction: column; align-items: center;">
            <h1>違規車輛</h1>
            <button onclick="updateImage()">取得當前</button>
            <img id="image" src="/image">
        </div>
    </body>
    </html>
    """
    return html_content


def generate_frames():
    while not isShutdown:
        if len(plates) == 0:
            continue
        time.sleep(0.1)
        ocr_plates = None
        ocr_plates = get_all_plates([plate.plate for plate in plates])
        if ocr_plates is None:
            continue
        # ####
        # max_width = max([plate.plate.shape[1] for plate in plates])
        # resized_images = [
        #     cv2.resize(plate.plate, (max_width, plate.plate.shape[0])) for plate in plates
        # ]
        # # 垂直拼接圖像
        # vertical_concat = cv2.vconcat(resized_images)
        # ####

        (flag, encodedImage) = cv2.imencode(".jpg", ocr_plates)
        if not flag:
            continue
        yield (
            b'--frame\r\n' b'Content-Type: image/jpeg\r\n\r\n' + bytearray(encodedImage) + b'\r\n'
        )
        # 等待一段時間再生成下一個圖像帧


@app.get("/image")
async def image():
    return StreamingResponse(
        generate_frames(), media_type="multipart/x-mixed-replace;boundary=frame"
    )


@app.get("/update")
async def update():
    for client in connected_clients:
        try:
            # seconds from 1970-01-01 00:00:00
            current_time = time.time()
            print("update")
            print(current_time)
            await client.send_json({"time": current_time})
        except Exception as e:
            print(e)
    return {"message": "update"}
    # func setInitPosition(uiImage:UIImage){
    #     // post image to /initPlateCam
    #     let data = uiImage.jpegData(compressionQuality: 1)
    #     let base64 = data?.base64EncodedString()
    #     let url = URL(string: "http://192.168.1.131/initPlateCam")
    #     var request = URLRequest(url: url!)
    #     request.httpMethod = "POST"
    #     let postString = "image=\(base64!)"
    #     request.httpBody = postString.data(using: .utf8)
    #     let task = URLSession.shared.dataTask(with: request) { data, response, error in
    #         guard let _ = data, error == nil else {
    #             print("error=\(String(describing: error))")
    #             return
    #         }
    #     }
    #     task.resume()
    #     print("setInitPosition")
    # }


@app.post("/initPlateCam/{cam}")
async def process_image(image: UploadFile = File(...), cam: str = ""):
    # 將上傳的圖片讀取為OpenCV影像
    image_data = await image.read()
    np_array = np.fromstring(image_data, np.uint8)
    cv_image = cv2.imdecode(np_array, cv2.IMREAD_COLOR)

    # 在此處執行您需要的處理邏輯，例如保存文件、進行分析等
    # 等比縮小到寬度為600
    width = 600
    height = int(width / cv_image.shape[1] * cv_image.shape[0])
    temp = cv_image.copy()
    temp = cv2.resize(temp, (width, height))

    # 顯示影像
    points = []

    def mouse_callback(event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            points.append([x, y])

    cv2.namedWindow("Image", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("Image", temp.shape[1], temp.shape[0])
    cv2.setMouseCallback("Image", mouse_callback)
    while len(points) < 4:
        for point in points:
            cv2.circle(temp, tuple(point), 5, (0, 0, 255), -1)
        cv2.polylines(temp, np.array([points]), True, (0, 0, 255), 2)
        cv2.imshow("Image", temp)
        key = cv2.waitKey(1) & 0xFF
        if key == ord("c"):
            break
    cv2.destroyAllWindows()

    print(points)
    # 還原到原始大小
    points = np.array(points) / (width / cv_image.shape[1])
    points = [[int(point[0]), int(point[1])] for point in points]
    print(points)
    cameras.add(cam, points, cv_image)
    cameras.getAllMatrixAndTransformedImage()

    return {"filename": image.filename}


@app.websocket("/event")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    connected_clients.add(websocket)
    try:
        while True:
            data = await websocket.receive_json()
            data = plate(**data)
            # clear plates
            plates.clear()
            await put_image(data)

            # print data instead of plate
            print("data")
            print(data.time)
            print(data.position)
            print(len(data.plate))

    except WebSocketDisconnect:
        connected_clients.remove(websocket)


async def put_image(data: plate):
    images_ = []
    for plate in data.plate:
        # decode base64
        encoded_image = plate
        decoded_image = base64.b64decode(encoded_image)
        nparr = np.frombuffer(decoded_image, dtype=np.uint8)
        image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        images_.append(image)
        plates.append(plateNumpy(time=data.time, plate=image, position=data.position))

    return {"message": "Image received and displayed"}


def show_image(images):
    for image in images:
        cv2.namedWindow("Image", cv2.WINDOW_NORMAL)
        cv2.imshow("Image", image)
        cv2.waitKey(100)
    cv2.destroyAllWindows()


if __name__ == "__main__":
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=80,
        reload=True,
    )
