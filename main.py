from asyncio import Lock
import os
from pathlib import Path
import random
import subprocess

from fastapi import FastAPI, Response, UploadFile
from fastapi.middleware.cors import CORSMiddleware

from torch_models.models import ResNetLSTM, ViTModel, ConvLSTM, LRCN


tmp_file_dir = "tmp/videos"
Path(tmp_file_dir).mkdir(parents=True, exist_ok=True)

lock = Lock()
images: list[bytes] = []


class EmptyModel:
    name = ""


current_model = EmptyModel()
labels_path = "torch_models/data/labels.txt"
with open(labels_path, "r") as file:
    labels = file.readlines()

app = FastAPI()
origins = [
    "http://localhost:5173",
    "http://127.0.0.1:5173",
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.post(
    path="/api/media-file/",
)
async def post_media_file(file: UploadFile):
    """
    Receive File, store to disk & return it
    """

    await lock.acquire()

    images.append(await file.read())
    filename: str | None = None

    got_video = False

    if len(images) >= 20:
        got_video = True

        # Convert list of images to mp4 video. Pipe images to ffmpeg
        filename = f"{random.randint(0, 100000)}.mp4"
        video_path = os.path.join(tmp_file_dir, filename)
        ffmpeg_cmd = f"ffmpeg -f image2pipe -vcodec mjpeg -r 24 -i - -vcodec mjpeg -b:v 2500k {video_path}"
        ffmpeg = subprocess.Popen(
            ffmpeg_cmd,
            shell=True,
            text=False,
            stdin=subprocess.PIPE,
            bufsize=-1,
        )
        stdin = ffmpeg.stdin
        assert stdin is not None
        for image in images:
            stdin.write(image)
        stdin.close()
        images.clear()
        print("Video converted")

    lock.release()

    return {
        "got_video": got_video,
        "id": filename,
    }


@app.get(path="/api/models/")
async def models():
    return [
        {
            "id": "resnet-lstm",
            "name": "ResNet LSTM",
        },
        {
            "id": "vit-model",
            "name": "ViT Model",
        },
        {
            "id": "conv-lstm",
            "name": "Conv LSTM",
        },
        {
            "id": "lrcn",
            "name": "LRCN",
        },
    ]


@app.post(
    path="/api/process/",
)
async def process_file(id: str, model: str):
    global current_model
    if current_model.name != model:
        if model == "resnet-lstm":
            m = ResNetLSTM()
        elif model == "vit-model":
            m = ViTModel()
        elif model == "conv-lstm":
            m = ConvLSTM()
        elif model == "lrcn":
            m = LRCN()
        else:
            return Response(
                {
                    "detail": f"No model with name {model} exists",
                },
                status_code=404,
            )

        m.load_pretrained()
        current_model = m

    data = current_model.prepare_data(os.path.join(tmp_file_dir, id))
    result = current_model.predict(data)

    # Delete video after processing
    os.remove(os.path.join(tmp_file_dir, id))

    return {
        "result": labels[result].replace("\n", ""),
    }
