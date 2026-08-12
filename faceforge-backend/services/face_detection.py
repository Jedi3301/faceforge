import asyncio
import sys
import time
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
import onnxruntime as ort

from core.config import settings


@dataclass
class FaceDetection:
    bbox: np.ndarray
    confidence: float
    landmarks: np.ndarray


class SCRFDDetector:
    """
    SCRFD-10G-BNKPS face detector.

    Outputs:
        - Bounding box
        - Detection confidence
        - 5 facial landmarks
    """

    def __init__(
        self,
        model_path: str,
        input_size: tuple[int, int] = (640, 640),
        confidence_threshold: float = 0.40,
        nms_threshold: float = 0.40,
    ):
        self.input_size = input_size
        self.confidence_threshold = confidence_threshold
        self.nms_threshold = nms_threshold

        # Load CUDA dependencies bundled with the environment.
        try:
            ort.preload_dlls(directory="")
        except AttributeError:
            pass

        self.session = ort.InferenceSession(
            model_path,
            providers=[
                "CUDAExecutionProvider",
                "CPUExecutionProvider",
            ],
        )

        self.input_name = self.session.get_inputs()[0].name

        print("SCRFD providers:")
        print(self.session.get_providers())

        print(f"Input name: {self.input_name}")
        print(f"Input shape: {self.session.get_inputs()[0].shape}")

    def _preprocess(
        self,
        image: np.ndarray,
    ) -> tuple[np.ndarray, float]:
        """
        Resize image while preserving aspect ratio.

        The resized image is placed in the top-left corner of a
        640x640 canvas.
        """

        original_height, original_width = image.shape[:2]

        target_width, target_height = self.input_size

        scale = min(
            target_width / original_width,
            target_height / original_height,
        )

        resized_width = int(original_width * scale)
        resized_height = int(original_height * scale)

        resized = cv2.resize(
            image,
            (resized_width, resized_height),
            interpolation=cv2.INTER_LINEAR,
        )

        canvas = np.zeros(
            (target_height, target_width, 3),
            dtype=np.uint8,
        )

        canvas[
            :resized_height,
            :resized_width,
        ] = resized

        # OpenCV image is BGR.
        # SCRFD expects RGB.
        canvas = cv2.cvtColor(
            canvas,
            cv2.COLOR_BGR2RGB,
        )

        canvas = canvas.astype(np.float32)

        # SCRFD normalization.
        canvas = (
            canvas - 127.5
        ) / 128.0

        # HWC -> CHW
        canvas = np.transpose(
            canvas,
            (2, 0, 1),
        )

        # Add batch dimension.
        canvas = np.expand_dims(
            canvas,
            axis=0,
        )

        # Ensure contiguous memory.
        canvas = np.ascontiguousarray(
            canvas,
            dtype=np.float32,
        )

        return canvas, scale

    def detect(
        self,
        image: np.ndarray,
    ) -> list[FaceDetection]:
        """
        Detect all faces in an image.
        """

        if image is None:
            raise ValueError("Image cannot be None.")

        input_tensor, scale = self._preprocess(
            image
        )

        outputs = self.session.run(
            None,
            {
                self.input_name: input_tensor,
            },
        )

        return self._postprocess(
            outputs,
            scale,
            image.shape[:2],
        )

    def _postprocess(
        self,
        outputs: list[np.ndarray],
        scale: float,
        original_shape: tuple[int, int],
    ) -> list[FaceDetection]:

        original_height, original_width = original_shape

        # ---------------------------------------------------------
        # Output layout of scrfd_10g_bnkps.onnx:
        #
        # 0, 1, 2 -> scores
        # 3, 4, 5 -> bounding boxes
        # 6, 7, 8 -> landmarks
        # ---------------------------------------------------------

        scores_outputs = outputs[0:3]
        bbox_outputs = outputs[3:6]
        landmark_outputs = outputs[6:9]

        # SCRFD feature-map strides.
        strides = [8, 16, 32]

        all_boxes = []
        all_scores = []
        all_landmarks = []

        for stride, scores, bbox_preds, landmark_preds in zip(
            strides,
            scores_outputs,
            bbox_outputs,
            landmark_outputs,
        ):

            scores = scores.reshape(-1)
            bbox_preds = bbox_preds.reshape(-1, 4)
            landmark_preds = landmark_preds.reshape(-1, 10)

            # Feature-map dimensions.
            feature_height = (
                self.input_size[1] // stride
            )

            feature_width = (
                self.input_size[0] // stride
            )

            # SCRFD-10G uses 2 anchors per location.
            num_anchors = 2

            # Generate anchor centers.
            grid_y, grid_x = np.mgrid[
                :feature_height,
                :feature_width,
            ]

            anchor_centers = np.stack(
                [
                    grid_x,
                    grid_y,
                ],
                axis=-1,
            ).reshape(-1, 2)

            anchor_centers = (
                anchor_centers.astype(np.float32)
                * stride
            )

            anchor_centers = np.repeat(
                anchor_centers,
                num_anchors,
                axis=0,
            )

            # Sanity check against model output.
            if len(anchor_centers) != len(scores):
                raise RuntimeError(
                    f"Anchor/output mismatch at stride {stride}: "
                    f"{len(anchor_centers)} anchors vs "
                    f"{len(scores)} predictions"
                )

            # -----------------------------------------------------
            # Confidence filtering
            # -----------------------------------------------------

            keep = (
                scores
                >= self.confidence_threshold
            )

            if not np.any(keep):
                continue

            scores = scores[keep]
            bbox_preds = bbox_preds[keep]
            landmark_preds = landmark_preds[keep]
            anchor_centers = anchor_centers[keep]

            # -----------------------------------------------------
            # Decode bounding boxes
            #
            # SCRFD bbox format:
            #
            # [distance_left,
            #  distance_top,
            #  distance_right,
            #  distance_bottom]
            # -----------------------------------------------------

            boxes = np.zeros_like(
                bbox_preds,
                dtype=np.float32,
            )

            boxes[:, 0] = (
                anchor_centers[:, 0]
                - bbox_preds[:, 0] * stride
            )

            boxes[:, 1] = (
                anchor_centers[:, 1]
                - bbox_preds[:, 1] * stride
            )

            boxes[:, 2] = (
                anchor_centers[:, 0]
                + bbox_preds[:, 2] * stride
            )

            boxes[:, 3] = (
                anchor_centers[:, 1]
                + bbox_preds[:, 3] * stride
            )

            # -----------------------------------------------------
            # Decode 5 landmarks
            #
            # landmark_preds contains:
            #
            # x1, y1,
            # x2, y2,
            # x3, y3,
            # x4, y4,
            # x5, y5
            # -----------------------------------------------------

            landmarks = np.zeros_like(
                landmark_preds,
                dtype=np.float32,
            )

            for point in range(5):

                landmarks[:, point * 2] = (
                    anchor_centers[:, 0]
                    + landmark_preds[:, point * 2]
                    * stride
                )

                landmarks[:, point * 2 + 1] = (
                    anchor_centers[:, 1]
                    + landmark_preds[:, point * 2 + 1]
                    * stride
                )

            # -----------------------------------------------------
            # Convert from model coordinates back to original
            # image coordinates.
            # -----------------------------------------------------

            boxes /= scale
            landmarks /= scale

            # Clip bounding boxes to image.
            boxes[:, 0] = np.clip(
                boxes[:, 0],
                0,
                original_width - 1,
            )

            boxes[:, 1] = np.clip(
                boxes[:, 1],
                0,
                original_height - 1,
            )

            boxes[:, 2] = np.clip(
                boxes[:, 2],
                0,
                original_width - 1,
            )

            boxes[:, 3] = np.clip(
                boxes[:, 3],
                0,
                original_height - 1,
            )

            all_boxes.append(boxes)
            all_scores.append(scores)
            all_landmarks.append(landmarks)

        if not all_boxes:
            return []

        # Combine the three feature levels.
        boxes = np.concatenate(
            all_boxes,
            axis=0,
        )

        scores = np.concatenate(
            all_scores,
            axis=0,
        )

        landmarks = np.concatenate(
            all_landmarks,
            axis=0,
        )

        # ---------------------------------------------------------
        # NMS
        # ---------------------------------------------------------

        nms_boxes = []

        for x1, y1, x2, y2 in boxes:

            nms_boxes.append(
                [
                    float(x1),
                    float(y1),
                    float(x2 - x1),
                    float(y2 - y1),
                ]
            )

        indices = cv2.dnn.NMSBoxes(
            nms_boxes,
            scores.tolist(),
            self.confidence_threshold,
            self.nms_threshold,
        )

        if len(indices) == 0:
            return []

        indices = np.asarray(
            indices
        ).reshape(-1)

        detections = []

        for index in indices:

            detections.append(
                FaceDetection(
                    bbox=boxes[index],
                    confidence=float(
                        scores[index]
                    ),
                    landmarks=landmarks[index].reshape(
                        5,
                        2,
                    ),
                )
            )

        # Highest confidence first.
        detections.sort(
            key=lambda detection: detection.confidence,
            reverse=True,
        )

        return detections


class FaceDetectionService:
    """
    Production-grade face detection service wrapping SCRFD-10G-BNKPS model.
    Thread-safe, singleton-friendly, with warm-up and lifecycle management.
    """

    def __init__(
        self,
        model_path: str,
        input_size: tuple[int, int] = (640, 640),
        confidence_threshold: float = 0.40,
        nms_threshold: float = 0.40,
    ):
        self.model_path = model_path
        self.input_size = input_size
        self.confidence_threshold = confidence_threshold
        self.nms_threshold = nms_threshold

        self._session = None
        self._detector = None

    def _load_model(self):
        """Load the ONNX model and initialize SCRFDDetector."""
        if self._detector is not None:
            return

        self._detector = SCRFDDetector(
            model_path=self.model_path,
            input_size=self.input_size,
            confidence_threshold=self.confidence_threshold,
            nms_threshold=self.nms_threshold,
        )

    async def warm_up(self):
        """Run a single inference with a dummy image to prime CUDA/CPP extensions."""
        if self._detector is None:
            self._load_model()

        # Create a dummy 640x640 RGB image (black)
        dummy_image = np.zeros(
            (self.input_size[1], self.input_size[0], 3), dtype=np.uint8
        )

        # Run detect in a thread to avoid blocking
        loop = asyncio.get_event_loop()
        await loop.run_in_executor(None, self._detector.detect, dummy_image)

    def detect(self, image: np.ndarray) -> list[FaceDetection]:
        """
        Detect faces in an image.

        Args:
            image: BGR numpy array (H, W, 3) as captured by OpenCV imread.

        Returns:
            List of FaceDetection objects, sorted by confidence (highest first).
        """
        if self._detector is None:
            self._load_model()

        # Run inference - SCRFDDetector.detect is synchronous
        return self._detector.detect(image)

    def shutdown(self):
        """Release resources. ONNX sessions are read-only after init, but good practice."""
        self._detector = None
        self._session = None