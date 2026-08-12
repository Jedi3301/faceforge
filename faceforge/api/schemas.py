import base64
import uuid
from io import BytesIO
from typing import List, Optional

import numpy as np
from fastapi import UploadFile
from PIL import Image
from pydantic import BaseModel, Field


class LandmarkModel(BaseModel):
    x: float = Field(..., description="X coordinate of facial landmark")
    y: float = Field(..., description="Y coordinate of facial landmark")


class BBoxModel(BaseModel):
    x1: float = Field(..., description="Top-left X coordinate of bounding box")
    y1: float = Field(..., description="Top-left Y coordinate of bounding box")
    x2: float = Field(..., description="Bottom-right X coordinate of bounding box")
    y2: float = Field(..., description="Bottom-right Y coordinate of bounding box")


class FaceDetectionModel(BaseModel):
    bbox: BBoxModel = Field(..., description="Bounding box coordinates")
    confidence: float = Field(
        ..., ge=0.0, le=1.0, description="Detection confidence score"
    )
    landmarks: List[LandmarkModel] = Field(
        default_factory=list,
        description="5 facial landmarks (eyes, nose, mouth corners)",
    )


class DetectRequest(BaseModel):
    image: Optional[UploadFile] = Field(
        None, description="Uploaded image file (multipart/form-data)"
    )
    image_base64: Optional[str] = Field(
        None, description="Base64-encoded image"
    )


class DetectResponse(BaseModel):
    request_id: str = Field(
        default_factory=lambda: str(uuid.uuid4()),
        description="Unique request identifier",
    )
    detections: List[FaceDetectionModel] = Field(
        default_factory=list,
        description="List of detected faces, sorted by confidence (highest first)",
    )
    processing_time_ms: int = Field(
        ..., description="Time spent in inference + post-processing (ms)"
    )