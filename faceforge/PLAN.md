# FaceForge - Face Detection Service

## Project Overview

FaceForge is a face detection service designed for large-scale events (weddings, anniversaries, etc.) where many photos are taken. The service provides a scalable API to detect faces accurately from crowd images, enabling downstream face recognition (Layer 2) and person-based photo search.

## Architecture

### 2-Layer Design

1. **Layer 1 - Face Detection** (Current): SCRFD-10G-BNKPS ONNX model wrapped in a production-grade FastAPI service
2. **Layer 2 - Face Recognition** (Future): Face embeddings generation and matching

### Service Architecture

```
faceforge/
├── app.py                  # FastAPI entry point, lifespan, middleware
├── core/
│   ├── __init__.py
│   └── config.py           # Pydantic Settings with env var configuration
├── services/
│   ├── __init__.py
│   └── face_detection.py   # SCRFDDetector + FaceDetectionService
├── api/
│   ├── __init__.py
│   ├── schemas.py          # Pydantic request/response models
│   └── routes/
│       ├── __init__.py
│       └── detect.py       # FastAPI endpoints
├── models/
│   └── scrfd_10g_bnkps.onnx # ONNX face detection model
├── images/
│   └── crowd.jpg            # Test image
└── output/                  # Output directory
```

## Current Configuration

| Parameter | Value | Description |
|-----------|-------|-------------|
| `confidence_threshold` | **0.40** | Minimum detection confidence (kept at 40s for clean faces) |
| `nms_threshold` | **0.40** | Non-Maximum Suppression threshold |
| `input_size` | **(640, 640)** | Model input dimensions |

## API Endpoints

### GET /
Root endpoint returning service info.

### GET /v1/health
Readiness/liveness probe.

**Response:**
```json
{"status": "ok"}
```

### POST /v1/detect
Detect faces in an image.

**Request (multipart/form-data):**
```
image: <uploaded image file>
```

**OR Request (JSON):**
```json
{"image_base64": "<base64-encoded-image>"}
```

**Response:**
```json
{
  "request_id": "uuid-string",
  "detections": [
    {
      "bbox": {"x1": 100.0, "y1": 150.0, "x2": 200.0, "y2": 250.0},
      "confidence": 0.87,
      "landmarks": [
        {"x": 110.0, "y": 160.0},
        {"x": 190.0, "y": 160.0},
        {"x": 150.0, "y": 200.0},
        {"x": 120.0, "y": 210.0},
        {"x": 180.0, "y": 210.0}
      ]
    }
  ],
  "processing_time_ms": 243
}
```

## Development & Deployment

### Prerequisites
- Python 3.13+
- uv package manager
- ONNX model file (scrfd_10g_bnkps.onnx)

### Running Locally

```bash
# Install dependencies
uv sync

# Start the service
cd faceforge
python -m uvicorn app:app --host 0.0.0.0 --port 8000
```

### Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `FACEFORGE_CONFIDENCE_THRESHOLD` | `0.40` | Minimum detection confidence |
| `FACEFORGE_NMS_THRESHOLD` | `0.40` | NMS overlap threshold |
| `FACEFORGE_INPUT_SIZE` | `640,640` | Model input dimensions |
| `FACEFORGE_HOST` | `0.0.0.0` | Server bind address |
| `FACEFORGE_PORT` | `8000` | Server port |
| `FACEFORGE_LOG_LEVEL` | `INFO` | Logging level |

## Implementation Log

### Phase 1: Project Setup & Face Detection (Complete)

#### [x] Step 1: Original Implementation
- Created standalone face detection script using SCRFD-10G-BNKPS ONNX model
- Components: `detector.py` (SCRFDDetector), `main.py` (CLI runner)
- Parameters: confidence_threshold=0.30, nms_threshold=0.40

#### [x] Step 2: Parameter Adjustment
- Adjusted `confidence_threshold` from 0.30 to **0.40** for cleaner face detection
- Confirmed threshold setting suitable for event photo crowds

#### [x] Step 3: Production Service Restructure
- Moved detector code from `face-recognition/src/` into `faceforge/services/face_detection.py`
- Created FastAPI application with clean architecture:
  - `core/config.py` - Pydantic settings management
  - `services/face_detection.py` - SCRFDDetector + FaceDetectionService wrapper
  - `api/schemas.py` - Pydantic request/response models
  - `api/routes/detect.py` - FastAPI endpoints
  - `app.py` - Application factory with lifespan, middleware, error handling
- Moved ONNX model to `faceforge/models/`
- Moved test images to `faceforge/images/`
- Removed old `face-recognition/` directory

#### [x] Step 4: Testing & Validation
- Service started successfully with CUDA fallback to CPU
- Health endpoint responding: `GET /v1/health` → `{"status": "ok"}`
- Face detection working: 58 faces detected in crowd.jpg at 0.40 threshold
- Processing time: ~243ms per 6MP image
- Proper error handling for invalid inputs
- X-Request-Id tracing for request correlation

#### [x] Step 5: Face Cropping Integration (Layer 1 Enhancement)
- Modified `/v1/detect` endpoint to automatically crop bounding box regions from the source image.
- Implemented isolated storage: each request creates a unique folder (`output/<request_id>/`).
- Saved individually cropped faces directly to disk (`face_0.jpg`, etc.) for Phase 2 readiness.

#### [x] Step 6: Database Integration (PostgreSQL & Alembic)
- Added `sqlalchemy`, `asyncpg`, and `alembic` for persistent request/crop tracking.
- Designed `RequestRecord` and `FaceCropRecord` tables using declarative SQLAlchemy models.
- Initialized Alembic for schema version control and ran the `Init Tables` migration against a local Docker Postgres instance.
- Integrated `AsyncSession` into `/v1/detect` to atomically save crop paths, bounding boxes, and processing times.

### Phase 2: Face Embeddings & Recognition (Planned)
- Add face alignment using detected landmarks
- Integrate face embedding model (FaceNet/ArcFace)
- Build face database for person identification
- Create search API for finding photos with specific people

### Phase 3: Web Frontend & User Experience (Planned)
- Photo upload interface for events
- Selfie upload for personal photo search
- Gallery view with face-tagged photos
- Sharing links for event attendees

## Running

```bash
cd faceforge
python -m uvicorn app:app --host 0.0.0.0 --port 8000
```

API Documentation: http://localhost:8000/api/docs