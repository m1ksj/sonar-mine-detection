from __future__ import annotations

import os
from functools import lru_cache
from pathlib import Path

from fastapi import FastAPI, File, HTTPException, Query, UploadFile
from PIL import Image, UnidentifiedImageError
from pydantic import BaseModel
from ultralytics import YOLO


CLASS_NAMES = {0: "MILCO", 1: "NOMBO"}
DEFAULT_MODEL_PATH = Path("models/final/yolo26n_best.pt")


class Detection(BaseModel):
    class_id: int
    class_name: str
    confidence: float
    box_xyxy: list[float]


class ImageInfo(BaseModel):
    filename: str | None
    width: int
    height: int


class PredictionResponse(BaseModel):
    model: str
    image: ImageInfo
    confidence_threshold: float
    detections_count: int
    detections: list[Detection]


app = FastAPI(
    title="Sonar Mine Detection API",
    description=(
        "Local API for YOLO26n side-scan sonar object detection."
    ),
    version="1.0.0",
)


def get_model_path() -> Path:
    return Path(os.getenv("SONAR_MODEL_PATH", DEFAULT_MODEL_PATH))


@lru_cache(maxsize=1)
def load_model() -> YOLO:
    model_path = get_model_path()

    if not model_path.exists():
        raise FileNotFoundError(
            f"Model not found at {model_path}. Place it at "
            "models/final/yolo26n_best.pt or set SONAR_MODEL_PATH."
        )

    return YOLO(str(model_path))


@app.get("/health")
def health() -> dict:
    model_path = get_model_path()

    return {
        "status": "ok",
        "model_path": str(model_path),
        "model_available": model_path.exists(),
    }


@app.get("/labels")
def labels() -> dict:
    return {"labels": CLASS_NAMES}


@app.post("/predict", response_model=PredictionResponse)
async def predict(
    file: UploadFile = File(
        ...,
        description="Input sonar image as JPG or PNG.",
    ),
    confidence: float = Query(
        0.25,
        ge=0.0,
        le=1.0,
    ),
    iou: float = Query(
        0.70,
        ge=0.0,
        le=1.0,
    ),
    device: str = Query("cpu"),
) -> PredictionResponse:
    if file.content_type is None:
        raise HTTPException(
            status_code=415,
            detail="Please upload an image file.",
        )

    if not file.content_type.startswith("image/"):
        raise HTTPException(
            status_code=415,
            detail="Please upload an image file.",
        )

    try:
        image = Image.open(file.file).convert("RGB")
    except UnidentifiedImageError as exc:
        raise HTTPException(
            status_code=400,
            detail="Could not read image.",
        ) from exc

    try:
        model = load_model()
    except FileNotFoundError as exc:
        raise HTTPException(
            status_code=500,
            detail=str(exc),
        ) from exc

    results = model.predict(
        source=image,
        conf=confidence,
        iou=iou,
        imgsz=640,
        device=device,
        verbose=False,
    )

    detections = []
    boxes = results[0].boxes

    if boxes is not None:
        for box in boxes:
            class_id = int(box.cls[0])
            xyxy = [float(x) for x in box.xyxy[0].tolist()]

            detections.append(
                Detection(
                    class_id=class_id,
                    class_name=CLASS_NAMES.get(class_id, str(class_id)),
                    confidence=float(box.conf[0]),
                    box_xyxy=xyxy,
                )
            )

    image_info = ImageInfo(
        filename=file.filename,
        width=image.width,
        height=image.height,
    )

    return PredictionResponse(
        model="YOLO26n",
        image=image_info,
        confidence_threshold=confidence,
        detections_count=len(detections),
        detections=detections,
    )
