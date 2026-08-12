import logging
import sys
import uuid
from pathlib import Path
from contextlib import asynccontextmanager

from fastapi import FastAPI, Request, Response
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.gzip import GZipMiddleware
from fastapi.responses import JSONResponse

from core.config import settings
from services.face_detection import FaceDetectionService
from api.schemas import DetectResponse
from api.routes.detect import router as detect_router

# ---------------------------------------------------------------------------
# Logging setup — simple format without request_id dependency
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=getattr(logging, settings.log_level.upper(), logging.INFO),
    format="%(asctime)s | %(name)s | %(levelname)s | %(message)s",
    stream=sys.stdout,
)
logger = logging.getLogger("faceforge")


# ---------------------------------------------------------------------------
# Lifespan: startup / shutdown
# ---------------------------------------------------------------------------
@asynccontextmanager
async def lifespan(app: FastAPI):
    # --- Startup ---
    logger.info("Starting FaceForge Detection Service...")

    # Initialise the model service and store on app state
    detector = FaceDetectionService(
        model_path=str(settings.model_path),
        input_size=settings.input_size,
        confidence_threshold=settings.confidence_threshold,
        nms_threshold=settings.nms_threshold,
    )

    # Warm-up inference (prime CUDA / C++ extensions)
    try:
        await detector.warm_up()
        logger.info("Warm-up inference completed successfully.")
    except Exception as e:
        logger.warning(f"Warm-up inference failed (may be CPU-only): {e}")

    app.state.detector = detector
    logger.info("FaceForge Detection Service started.")

    yield  # <-- app runs here

    # --- Shutdown ---
    logger.info("Shutting down FaceForge Detection Service...")
    app.state.detector.shutdown()
    logger.info("Resources released.")


# ---------------------------------------------------------------------------
# FastAPI application factory
# ---------------------------------------------------------------------------
def create_app() -> FastAPI:
    """
    Create and configure the FastAPI application.
    """
    app = FastAPI(
        title="FaceForge Detection Service",
        version="1.0.0",
        description="Production-grade face detection Layer 1 service for event photos",
        lifespan=lifespan,
        docs_url="/api/docs",
        redoc_url="/api/redoc",
    )

    # --- Middleware ---

    # CORS – restrict in production; allow all for dev convenience
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],  # <- tighten for prod (e.g., ["http://localhost:3000"])
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # GZip compression for responses > 1 KB
    app.add_middleware(GZipMiddleware, minimum_size=1000)

    # --- Request-ID middleware for tracing ---
    @app.middleware("http")
    async def add_request_id(request: Request, call_next):
        request_id = request.headers.get("X-Request-Id", None)
        if not request_id:
            request_id = str(uuid.uuid4())
        request.state.request_id = request_id

        response: Response = await call_next(request)
        response.headers["X-Request-Id"] = request_id
        return response

    # --- Exception handler for consistent error format ---

    @app.exception_handler(Exception)
    async def global_exception_handler(request: Request, exc: Exception):
        request_id = getattr(request.state, "request_id", "unknown")
        logger.exception(
            f"Unhandled exception [{request_id}]: {exc}",
            extra={"request_id": request_id},
        )
        error_response = {
            "detail": "Internal server error",
            "request_id": request_id,
        }
        return JSONResponse(
            status_code=500,
            content=error_response,
        )

    # --- Register routes ---

    app.include_router(detect_router)

    @app.get("/", include_in_schema=False)
    async def root() -> dict:
        return {"message": "FaceForge Detection Service API", "version": "1.0.0"}

    return app


# ---------------------------------------------------------------------------
# Module-level app (for `uvicorn faceforge.app:app`)
# ---------------------------------------------------------------------------
app = create_app()