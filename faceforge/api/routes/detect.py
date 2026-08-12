import asyncio
import base64
import time
import uuid
from io import BytesIO
from pathlib import Path
from typing import List

import numpy as np
import cv2
from fastapi import APIRouter, Depends, HTTPException, Request, UploadFile, status
from fastapi.responses import JSONResponse
from PIL import Image
from sqlalchemy.ext.asyncio import AsyncSession

from core.config import settings
from services.face_detection import FaceDetectionService, FaceDetection
from db.session import get_db
from db.models import RequestRecord, FaceCropRecord
from api.schemas import (
    DetectRequest,
    DetectResponse,
    FaceDetectionModel,
    LandmarkModel,
    BBoxModel,
)

router = APIRouter(prefix="/v1", tags=["face-detection"])


def get_detector(request: Request) -> FaceDetectionService:
    """Dependency: retrieve the detector from app state."""
    detector = getattr(request.app.state, "detector", None)
    if detector is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Face detection service not initialized",
        )
    return detector


def _pil_to_np(pil_image: Image.Image) -> np.ndarray:
    """Convert PIL Image to BGR numpy array (OpenCV format)."""
    img = np.array(pil_image.convert("RGB"))
    img_bgr = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    return img_bgr


def _parse_image_file(file: UploadFile) -> np.ndarray:
    """Read uploaded file into BGR numpy array."""
    content = file.file.read()
    pil_image = Image.open(BytesIO(content)).convert("RGB")
    return _pil_to_np(pil_image)


def _parse_base64_image(b64_str: str) -> np.ndarray:
    """Decode data URI or pure base64 string into BGR numpy array."""
    # Strip data URI prefix if present
    if b64_str.startswith("data:"):
        header, b64_data = b64_str.split(",", 1)
    else:
        b64_data = b64_str

    decoded = base64.b64decode(b64_data)
    pil_image = Image.open(BytesIO(decoded)).convert("RGB")
    return _pil_to_np(pil_image)


@router.get("/health", summary="Health check", tags=["general"])
async def health_check(request: Request) -> JSONResponse:
    """Readiness/liveness probe. Returns service status."""
    return JSONResponse(content={"status": "ok"}, status_code=200)


@router.post(
    "/detect",
    response_model=DetectResponse,
    summary="Detect faces in image",
    tags=["face-detection"],
)
async def detect_faces(
    request: Request,
    payload: DetectRequest = Depends(),
    detector: FaceDetectionService = Depends(get_detector),
    db: AsyncSession = Depends(get_db),
):
    """
    Detect faces in an uploaded image.

    Accepts either:
    - `image`: multipart/form-data file upload
    - `image_base64`: JSON body with base64-encoded image

    Returns detection results with bounding boxes, confidence scores,
    and 5 facial landmarks per face.
    """
    start_time = time.time()

    # --- Resolve image ---
    img_np: np.ndarray
    try:
        if payload.image is not None:
            img_np = _parse_image_file(payload.image)
        elif payload.image_base64 is not None:
            img_np = _parse_base64_image(payload.image_base64)
        else:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="No image provided",
            )
    except Exception as exc:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Failed to parse image: {exc}",
        )

    # --- Validate image dimensions ---
    h, w = img_np.shape[:2]
    if h < 10 or w < 10:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Image too small for face detection",
        )

    # --- Run detection in thread so event loop is not blocked ---
    loop = asyncio.get_event_loop()
    detect_start = time.time()
    detections: list = await loop.run_in_executor(
        None,
        detector.detect,
        img_np,
    )
    processing_ms = int((time.time() - detect_start) * 1000)

    # Generate request_id early so we can use it for the directory name
    request_id = request.headers.get("X-Request-Id", str(uuid.uuid4()))
    
    # Create output directory
    output_dir = Path(f"output/{request_id}")
    output_dir.mkdir(parents=True, exist_ok=True)

    db_faces = []

    # --- Convert internal entities to Pydantic models & save crops ---
    face_detections: List[FaceDetectionModel] = []
    for i, det in enumerate(detections):
        # det.bbox is np.ndarray [4] -> [x1, y1, x2, y2] in original image coords
        bbox_np = det.bbox.astype(float)
        # Landmarks: det.landmarks is np.ndarray (5, 2) -> [(x, y), ...]
        lms = det.landmarks

        bbox = BBoxModel(
            x1=float(bbox_np[0]),
            y1=float(bbox_np[1]),
            x2=float(bbox_np[2]),
            y2=float(bbox_np[3]),
        )

        landmarks: List[LandmarkModel] = []
        for lm in lms:
            landmarks.append(
                LandmarkModel(x=float(lm[0]), y=float(lm[1]))
            )

        face_detections.append(
            FaceDetectionModel(
                bbox=bbox,
                confidence=float(det.confidence),
                landmarks=landmarks,
            )
        )

        # Crop and save face
        x1, y1, x2, y2 = int(bbox.x1), int(bbox.y1), int(bbox.x2), int(bbox.y2)
        # Clamp to image bounds
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(w, x2), min(h, y2)
        
        if x2 > x1 and y2 > y1:
            file_path = str(output_dir / f"face_{i}.jpg")
            face_crop = img_np[y1:y2, x1:x2]
            cv2.imwrite(file_path, face_crop)
            
            db_faces.append(
                FaceCropRecord(
                    request_id=request_id,
                    bbox_x1=bbox.x1,
                    bbox_y1=bbox.y1,
                    bbox_x2=bbox.x2,
                    bbox_y2=bbox.y2,
                    confidence=float(det.confidence),
                    file_path=file_path
                )
            )

    # Sort by confidence descending (service already does this, but ensure)
    face_detections.sort(key=lambda d: d.confidence, reverse=True)

    total_ms = int((time.time() - start_time) * 1000)
    
    # Save to database
    db_req = RequestRecord(id=request_id, processing_time_ms=total_ms)
    db.add(db_req)
    db.add_all(db_faces)
    await db.commit()

    response = DetectResponse(
        request_id=request_id,
        detections=face_detections,
        processing_time_ms=total_ms,
    )

    # Add request ID to response headers
    return JSONResponse(
        content=response.model_dump(),
        status_code=200,
        headers={"X-Request-Id": response.request_id},
    )