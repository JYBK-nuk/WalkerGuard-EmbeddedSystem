import base64
from pprint import pprint
from fastapi import FastAPI, UploadFile, WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse
import uvicorn
import cv2
import numpy as np
from typing import List
import pydantic


app = FastAPI()


connected_clients = set()


class plate(pydantic.BaseModel):
    time: float
    plate: list[str]
    position: list[list[int]]


@app.get("/")
async def root():
    for client in connected_clients:
        try:
            await client.send_json({"event": "get"})
        except Exception as e:
            print(e)

    return {"message": "Hello World"}


@app.websocket("/event")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    connected_clients.add(websocket)
    try:
        while True:
            data = await websocket.receive_json()
            for plate_ in data:
                print(plate_.keys())
            data = [plate(**plate_) for plate_ in data]
            await put_image(data)

            pprint(data)

    except WebSocketDisconnect:
        connected_clients.remove(websocket)


async def put_image(data: list[plate]):
    images_ = []
    for tickData in data[-1:]:
        for plate in tickData.plate:
            # decode base64
            encoded_image = plate
            decoded_image = base64.b64decode(encoded_image)
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


if __name__ == "__main__":
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=80,
        reload=True,
    )
