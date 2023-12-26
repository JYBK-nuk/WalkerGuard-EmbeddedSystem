import base64
from fastapi import FastAPI, UploadFile, WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse
import uvicorn
import cv2
import numpy as np
from typing import List


app = FastAPI()


connected_clients = set()


@app.get("/")
async def root():
    for client in connected_clients:
        await client.send_json({"event": "get"})

    return {"message": "Hello World"}


@app.websocket("/event")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    connected_clients.add(websocket)
    try:
        while True:
            data = await websocket.receive_text()

    except WebSocketDisconnect:
        connected_clients.remove(websocket)


@app.put("/cam2")
async def put_image(images: List[UploadFile]):
    images_ = []
    for image in images:
        contents = await image.read()
        # 將Base64編碼的圖片解碼
        # 解碼Base64資料
        encoded_image = contents.split(b",")[1]

        decoded_image = base64.b64decode(encoded_image)

        # 將解碼後的圖片轉換為NumPy數組
        nparr = np.frombuffer(decoded_image, dtype=np.uint8)
        image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

        images_.append(image)

    show_image(images_)
    return {"message": "Image received and displayed"}


def show_image(images):
    for image in images:
        cv2.namedWindow("Image", cv2.WINDOW_NORMAL)
        cv2.imshow("Image", image)
        cv2.waitKey(0)
    cv2.destroyAllWindows() 


@app.get("/ios")
async def web():
    return FileResponse('template/CamInput.html')


if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=80, reload=True)
