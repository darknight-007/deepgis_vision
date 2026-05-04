#!/usr/bin/env python3
"""
Multi-Model AI Vision Node for ROS2

A flexible vision node that supports multiple AI backends:
- Grounding DINO (open-vocabulary detection)
- YOLOv8 (real-time object detection)
- SAM (Segment Anything Model)
- Custom models via plugin architecture

Based on DeepGIS-XR AI server architecture:
https://github.com/Earth-Innovation-Hub/deepgis-xr
"""

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy, DurabilityPolicy

from sensor_msgs.msg import Image, CompressedImage
from std_msgs.msg import String, Header

import cv2
import numpy as np
from cv_bridge import CvBridge, CvBridgeError

import os
import zlib
import requests
import base64
import json
import time
from threading import Lock, Thread
from queue import Queue, Empty
from datetime import datetime
from typing import List, Dict, Any, Optional, Tuple
from abc import ABC, abstractmethod
from dataclasses import dataclass, field, asdict
from enum import Enum


def _json_default(obj):
    """JSON encoder fallback for non-serializable types we emit (e.g. mask_rle bytes)."""
    if isinstance(obj, (bytes, bytearray)):
        return base64.b64encode(bytes(obj)).decode('ascii') if obj else ''
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj)
    raise TypeError(f"Object of type {type(obj).__name__} is not JSON serializable")


# Expanded BGR palette so per-camera shifts stay visually distinct (left vs right).
CLASS_COLOR_PALETTE: Tuple[Tuple[int, int, int], ...] = (
    (0, 255, 0), (0, 180, 255), (255, 128, 0), (255, 0, 255),
    (255, 255, 0), (0, 140, 255), (180, 105, 255), (128, 255, 128),
    (255, 80, 80), (80, 200, 240), (200, 200, 0), (0, 255, 190),
    (255, 0, 127), (127, 255, 0), (210, 0, 210), (0, 255, 255),
    (170, 120, 50), (100, 100, 255), (255, 180, 100), (100, 255, 180),
)


def _palette_color_bgr(class_label: str, palette_shift: int) -> Tuple[int, int, int]:
    """Stable color per label; `palette_shift` rotates the palette (different cameras/models)."""
    n = len(CLASS_COLOR_PALETTE)
    idx = (zlib.crc32(class_label.encode('utf-8')) % n + int(palette_shift)) % n
    return CLASS_COLOR_PALETTE[idx]


def _draw_dashed_line(img, p1, p2, color, thickness: int = 2, dash: int = 10, gap: int = 6):
    """Straight dashed segment."""
    x1, y1 = float(p1[0]), float(p1[1])
    x2, y2 = float(p2[0]), float(p2[1])
    dist = float(np.hypot(x2 - x1, y2 - y1))
    if dist < 1e-3:
        return
    ux, uy = (x2 - x1) / dist, (y2 - y1) / dist
    t = 0.0
    draw_on = True
    while t < dist:
        seg = dash if draw_on else gap
        t_next = min(t + seg, dist)
        if draw_on:
            cv2.line(
                img,
                (int(x1 + ux * t), int(y1 + uy * t)),
                (int(x1 + ux * t_next), int(y1 + uy * t_next)),
                color, thickness, cv2.LINE_AA,
            )
        t = t_next
        draw_on = not draw_on


def draw_detection_overlay(
    image: np.ndarray,
    detections: List[Dict],
    *,
    palette_shift: int = 0,
    latency_ms: float = 0.0,
    header_top: str = '',
    header_sub: str = '',
    dashed_boxes: bool = False,
    box_thickness: int = 2,
) -> np.ndarray:
    """
    Draw detection boxes + labels on a BGR image (shared by the ROS node and
    offline benchmarking). Bounding boxes must use *_norm fields (0–1 range).
    """
    h, w = image.shape[:2]

    for det in detections:
        label = det.get('class_name', 'unknown')
        conf = det.get('confidence', 0.0)
        color = _palette_color_bgr(label, palette_shift)

        x1 = int(det.get('bbox_x_norm', 0) * w)
        y1 = int(det.get('bbox_y_norm', 0) * h)
        x2 = int((det.get('bbox_x_norm', 0) + det.get('bbox_width_norm', 0)) * w)
        y2 = int((det.get('bbox_y_norm', 0) + det.get('bbox_height_norm', 0)) * h)

        if dashed_boxes:
            _draw_dashed_line(image, (x1, y1), (x2, y1), color, box_thickness)
            _draw_dashed_line(image, (x2, y1), (x2, y2), color, box_thickness)
            _draw_dashed_line(image, (x2, y2), (x1, y2), color, box_thickness)
            _draw_dashed_line(image, (x1, y2), (x1, y1), color, box_thickness)
        else:
            cv2.rectangle(image, (x1, y1), (x2, y2), color, box_thickness)

        mn = det.get('model_name', '')
        suffix = f" [{mn}]" if mn else ''
        text = f'{label}: {conf:.2f}{suffix}'
        (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        label_y = max(y1, th + 8)
        cv2.rectangle(image, (x1, label_y - th - 8), (x1 + tw + 4, label_y), color, -1)
        cv2.putText(
            image, text, (x1 + 2, label_y - 4),
            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA,
        )

    lines: List[str] = []
    if header_top.strip():
        lines.append(header_top)
    if header_sub.strip():
        lines.append(header_sub)
    elif latency_ms > 0:
        lines.append(
            f'latency {latency_ms:.1f} ms  ({1000.0 / max(latency_ms, 0.001):.0f} FPS)'
        )

    y0 = 22
    for i, ln in enumerate(lines):
        yy = y0 + i * 22
        cv2.putText(image, ln, (10, yy), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 3, cv2.LINE_AA)
        cv2.putText(image, ln, (10, yy), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 1, cv2.LINE_AA)

    return image


class ModelType(Enum):
    """Supported AI model types."""
    GROUNDING_DINO = "grounding_dino"
    YOLO = "yolo"
    SAM = "sam"
    CUSTOM = "custom"


@dataclass
class Detection:
    """Detection result container."""
    id: int = 0
    class_name: str = ""
    class_id: int = 0
    confidence: float = 0.0
    bbox_x: float = 0.0
    bbox_y: float = 0.0
    bbox_width: float = 0.0
    bbox_height: float = 0.0
    bbox_x_norm: float = 0.0
    bbox_y_norm: float = 0.0
    bbox_width_norm: float = 0.0
    bbox_height_norm: float = 0.0
    mask_rle: bytes = b""
    prompt: str = ""
    model_name: str = ""


@dataclass
class AIServerConfig:
    """Configuration for an AI server."""
    name: str
    url: str
    model_type: ModelType
    endpoint: str = "/api/predict"
    health_endpoint: str = "/health"
    api_key: str = ""
    timeout: float = 5.0
    priority: int = 0  # Lower = higher priority


class AIServerClient(ABC):
    """Abstract base class for AI server clients."""
    
    def __init__(self, config: AIServerConfig, logger):
        self.config = config
        self.logger = logger
        self.session = requests.Session()
        if config.api_key:
            self.session.headers.update({'Authorization': f'Bearer {config.api_key}'})
        self.is_healthy = False
        self.last_latency_ms = 0.0
    
    @abstractmethod
    def detect(self, image_base64: str, prompt: str, **kwargs) -> List[Detection]:
        """Perform detection on image."""
        pass

    def detect_image(self, cv_image: np.ndarray, prompt: str = "", **kwargs) -> List[Detection]:
        """Detect on a BGR cv2 image. Default impl JPEG-encodes and calls detect()."""
        encode_params = [cv2.IMWRITE_JPEG_QUALITY, kwargs.pop('jpeg_quality', 85)]
        ok, buffer = cv2.imencode('.jpg', cv_image, encode_params)
        if not ok:
            return []
        image_base64 = base64.b64encode(buffer).decode('utf-8')
        return self.detect(image_base64, prompt=prompt, **kwargs)

    def health_check(self) -> bool:
        """Check server health."""
        try:
            url = f"{self.config.url}{self.config.health_endpoint}"
            response = self.session.get(url, timeout=3.0)
            self.is_healthy = response.status_code == 200
            return self.is_healthy
        except Exception as e:
            self.logger.debug(f"Health check failed for {self.config.name}: {e}")
            self.is_healthy = False
            return False


class GroundingDINOClient(AIServerClient):
    """Client for Grounding DINO AI server."""
    
    def detect(
        self,
        image_base64: str,
        prompt: str,
        box_threshold: float = 0.35,
        text_threshold: float = 0.25,
        **kwargs
    ) -> List[Detection]:
        """
        Detect objects using Grounding DINO.
        
        API Format (multipart/form-data):
        - file: image binary
        - text_prompt: "person . car"
        - box_threshold: 0.35
        - text_threshold: 0.25
        
        Response: {predictions: {count, phrases, logits, boxes}, annotated_image}
        """
        start_time = time.time()
        
        try:
            # Convert base64 back to bytes for multipart
            import base64
            image_bytes = base64.b64decode(image_base64)
            
            files = {
                'file': ('image.jpg', image_bytes, 'image/jpeg')
            }
            data = {
                'text_prompt': prompt,
                'box_threshold': str(box_threshold),
                'text_threshold': str(text_threshold),
            }
            
            url = f"{self.config.url}{self.config.endpoint}"
            response = self.session.post(url, files=files, data=data, timeout=self.config.timeout)
            
            if response.status_code != 200:
                self.logger.warning(f"Grounding DINO returned {response.status_code}")
                return []
            
            result = response.json()
            self.last_latency_ms = (time.time() - start_time) * 1000
            
            # Parse new response format
            predictions = result.get('predictions', {})
            detections = []
            
            if predictions.get('count', 0) > 0:
                phrases = predictions.get('phrases', [])
                logits = predictions.get('logits', [])
                boxes = predictions.get('boxes', [])
                
                for i in range(len(phrases)):
                    box = boxes[i] if i < len(boxes) else [0, 0, 1, 1]
                    detections.append(Detection(
                        id=i,
                        class_name=phrases[i],
                        confidence=logits[i] if i < len(logits) else 0.0,
                        bbox_x_norm=box[0],
                        bbox_y_norm=box[1],
                        bbox_width_norm=box[2] - box[0],
                        bbox_height_norm=box[3] - box[1],
                        prompt=prompt,
                        model_name="grounding_dino",
                    ))
            
            return detections
            
        except Exception as e:
            self.logger.error(f"Grounding DINO detection error: {e}")
            return []


class YOLOClient(AIServerClient):
    """Client for YOLO AI server."""
    
    def detect(
        self,
        image_base64: str,
        prompt: str = "",
        confidence: float = 0.5,
        **kwargs
    ) -> List[Detection]:
        """
        Detect objects using YOLO.
        
        DeepGIS-XR API Format:
        Request: {image, confidence}
        Response: {detections: [{class, confidence, bbox}], inference_time}
        """
        start_time = time.time()
        
        payload = {
            'image': image_base64,
            'confidence': confidence,
        }
        
        try:
            url = f"{self.config.url}{self.config.endpoint}"
            response = self.session.post(url, json=payload, timeout=self.config.timeout)
            
            if response.status_code != 200:
                self.logger.warning(f"YOLO returned {response.status_code}")
                return []
            
            result = response.json()
            self.last_latency_ms = (time.time() - start_time) * 1000
            
            detections = []
            for i, det in enumerate(result.get('detections', [])):
                bbox = det.get('bbox', det.get('box', [0, 0, 1, 1]))
                detections.append(Detection(
                    id=i,
                    class_name=det.get('class', det.get('label', 'unknown')),
                    class_id=det.get('class_id', 0),
                    confidence=det.get('confidence', 0.0),
                    bbox_x_norm=bbox[0],
                    bbox_y_norm=bbox[1],
                    bbox_width_norm=bbox[2] - bbox[0] if len(bbox) == 4 else bbox[2],
                    bbox_height_norm=bbox[3] - bbox[1] if len(bbox) == 4 else bbox[3],
                    model_name="yolo",
                ))
            
            return detections
            
        except Exception as e:
            self.logger.error(f"YOLO detection error: {e}")
            return []


class SAMClient(AIServerClient):
    """Client for Segment Anything Model (SAM) server."""
    
    def detect(
        self,
        image_base64: str,
        prompt: str = "",
        points: List[Tuple[int, int]] = None,
        box: List[float] = None,
        **kwargs
    ) -> List[Detection]:
        """
        Perform segmentation using SAM.
        
        DeepGIS-XR API Format:
        Request: {image, points?, box?, auto_segment?}
        Response: {masks: [{mask_rle, score, area}]}
        """
        start_time = time.time()
        
        payload = {
            'image': image_base64,
            'auto_segment': True if not points and not box else False,
        }
        if points:
            payload['points'] = points
        if box:
            payload['box'] = box
        
        try:
            url = f"{self.config.url}{self.config.endpoint}"
            response = self.session.post(url, json=payload, timeout=self.config.timeout)
            
            if response.status_code != 200:
                self.logger.warning(f"SAM returned {response.status_code}")
                return []
            
            result = response.json()
            self.last_latency_ms = (time.time() - start_time) * 1000
            
            detections = []
            for i, mask in enumerate(result.get('masks', [])):
                detections.append(Detection(
                    id=i,
                    class_name="segment",
                    confidence=mask.get('score', 0.0),
                    mask_rle=mask.get('mask_rle', b''),
                    model_name="sam",
                ))
            
            return detections
            
        except Exception as e:
            self.logger.error(f"SAM segmentation error: {e}")
            return []


class CoralClient(AIServerClient):
    """
    On-device Google Coral Edge TPU detector (no HTTP).

    Loads a pre-compiled `*_edgetpu.tflite` SSD-style detector with built-in
    NMS post-processing (e.g. SSDLite MobileDet, SSD MobileNet v2). Inference
    runs in-process via pycoral; ROS callbacks hand cv_images straight in
    through `detect_image`, skipping the JPEG/base64 round-trip.
    """

    def __init__(
        self,
        config: AIServerConfig,
        logger,
        model_path: str,
        labels_path: str = '',
        score_threshold: float = 0.4,
        model_name_tag: str = 'coral',
    ):
        super().__init__(config, logger)
        try:
            from pycoral.utils.edgetpu import make_interpreter, list_edge_tpus
            from pycoral.utils.dataset import read_label_file
            from pycoral.adapters import common as coral_common
            from pycoral.adapters import detect as coral_detect
        except ImportError as e:
            raise RuntimeError(
                "pycoral / tflite_runtime not importable. Install via the "
                "feranick wheels (cp312 linux_x86_64) and `libedgetpu1-std` deb."
            ) from e

        if not model_path or not os.path.exists(model_path):
            raise FileNotFoundError(f"Coral model not found at {model_path!r}")

        tpus = list_edge_tpus()
        if not tpus:
            raise RuntimeError("No Edge TPU detected. Replug the Coral USB stick.")
        logger.info(f"CoralClient: {len(tpus)} Edge TPU(s) detected: {tpus}")

        self._common = coral_common
        self._detect = coral_detect
        self._interp = make_interpreter(model_path)
        self._interp.allocate_tensors()
        self._in_w, self._in_h = coral_common.input_size(self._interp)

        if labels_path and os.path.exists(labels_path):
            self.labels = read_label_file(labels_path)
        else:
            self.labels = {}
            if labels_path:
                logger.warning(f"CoralClient: labels file not found: {labels_path!r}")

        self.score_threshold = float(score_threshold)
        self._lock = Lock()
        self.is_healthy = True
        self.model_path = model_path
        self.model_name_tag = (model_name_tag or 'coral').strip()

        logger.info(
            f"CoralClient ready: model={os.path.basename(model_path)} "
            f"input={self._in_w}x{self._in_h} score_thr={self.score_threshold} "
            f"classes={len(self.labels)}"
        )

    def health_check(self) -> bool:
        return self.is_healthy

    def detect(self, image_base64: str, prompt: str = "", **kwargs) -> List[Detection]:
        """HTTP-style entry kept for API symmetry; decodes then delegates."""
        try:
            img_bytes = base64.b64decode(image_base64)
            np_arr = np.frombuffer(img_bytes, np.uint8)
            cv_image = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
            if cv_image is None:
                return []
            return self.detect_image(cv_image, prompt=prompt, **kwargs)
        except Exception as e:
            self.logger.error(f"Coral detect (base64 path) error: {e}")
            return []

    def detect_image(self, cv_image: np.ndarray, prompt: str = "", **kwargs) -> List[Detection]:
        if cv_image is None or cv_image.size == 0:
            return []

        score_thr = float(kwargs.get('confidence', self.score_threshold))

        h, w = cv_image.shape[:2]
        rgb = cv2.cvtColor(cv_image, cv2.COLOR_BGR2RGB)
        resized = cv2.resize(rgb, (self._in_w, self._in_h), interpolation=cv2.INTER_LINEAR)

        start = time.time()
        with self._lock:
            self._common.set_input(self._interp, resized)
            self._interp.invoke()
            objs = self._detect.get_objects(
                self._interp,
                score_threshold=score_thr,
                image_scale=(self._in_w / max(w, 1), self._in_h / max(h, 1)),
            )
        self.last_latency_ms = (time.time() - start) * 1000.0

        detections: List[Detection] = []
        for i, o in enumerate(objs):
            bb = o.bbox
            label = self.labels.get(o.id, str(o.id))
            xmin = max(0.0, float(bb.xmin))
            ymin = max(0.0, float(bb.ymin))
            xmax = min(float(w), float(bb.xmax))
            ymax = min(float(h), float(bb.ymax))
            bw = max(0.0, xmax - xmin)
            bh = max(0.0, ymax - ymin)
            detections.append(Detection(
                id=i,
                class_name=label,
                class_id=int(o.id),
                confidence=float(o.score),
                bbox_x=xmin,
                bbox_y=ymin,
                bbox_width=bw,
                bbox_height=bh,
                bbox_x_norm=xmin / max(w, 1),
                bbox_y_norm=ymin / max(h, 1),
                bbox_width_norm=bw / max(w, 1),
                bbox_height_norm=bh / max(h, 1),
                mask_rle=b'',
                model_name=self.model_name_tag,
            ))
        return detections


class AIVisionNode(Node):
    """
    Multi-model AI Vision Node for ROS2.
    
    Supports multiple AI backends with automatic failover:
    - Grounding DINO for open-vocabulary detection
    - YOLO for real-time detection
    - SAM for segmentation
    """

    def __init__(self):
        super().__init__('ai_vision_node')

        # Declare parameters
        self._declare_parameters()
        
        # Get parameters
        self._get_parameters()

        # Initialize components
        self.cv_bridge = CvBridge()
        self.data_lock = Lock()
        self.ai_clients: Dict[str, AIServerClient] = {}
        self.active_client: Optional[AIServerClient] = None
        
        # Statistics
        self.stats = {
            'frames_received': 0,
            'frames_processed': 0,
            'detections_total': 0,
            'avg_latency_ms': 0.0,
            'latency_samples': [],
        }
        
        # Rate limiting
        self.last_detection_time = 0.0
        self.min_detection_interval = 1.0 / self.max_fps
        
        # Async processing
        self.processing_queue: Queue = Queue(maxsize=self.queue_size)
        self.result_queue: Queue = Queue()
        self.shutdown_flag = False

        # Initialize AI clients
        self._init_clients()

        self._annotate_model_title = self._compose_annotate_title()

        # Setup ROS interfaces
        self._setup_ros_interfaces()

        # Start async processing
        if self.async_processing:
            self.processing_thread = Thread(target=self._processing_loop, daemon=True)
            self.processing_thread.start()
            self.result_timer = self.create_timer(0.01, self._process_results)

        # Health monitoring
        self.health_timer = self.create_timer(10.0, self._health_check_all)
        self.stats_timer = self.create_timer(5.0, self._publish_stats)

        # Initial health check
        self._health_check_all()

        self._log_startup()

    @staticmethod
    def _guess_model_title(model_path: str) -> str:
        base = os.path.basename(model_path or '').replace(
            '_edgetpu.tflite', '').replace('.tflite', '')
        return base or 'detector'

    def _compose_annotate_title(self) -> str:
        if self.coral_model_display_name:
            return self.coral_model_display_name
        if self.primary_server_type == 'coral':
            return self._guess_model_title(self.coral_model_path)
        return str(self.primary_server_type)

    def _declare_parameters(self):
        """Declare all node parameters."""
        # Server configuration
        self.declare_parameter('primary_server_url', 'http://localhost:5000')
        self.declare_parameter('primary_server_type', 'grounding_dino')
        self.declare_parameter('fallback_server_url', '')
        self.declare_parameter('fallback_server_type', 'yolo')
        self.declare_parameter('api_key', '')
        
        # Camera parameters
        self.declare_parameter('camera_topics', ['/stereo/right/image_raw'])
        self.declare_parameter('use_compressed', False)
        
        # Detection parameters
        self.declare_parameter('prompt', 'person . car . tree')
        self.declare_parameter('confidence_threshold', 0.3)
        self.declare_parameter('box_threshold', 0.35)
        self.declare_parameter('text_threshold', 0.25)
        
        # Performance parameters
        self.declare_parameter('max_fps', 10.0)
        self.declare_parameter('resize_width', 640)
        self.declare_parameter('resize_height', 480)
        self.declare_parameter('jpeg_quality', 85)
        self.declare_parameter('request_timeout', 5.0)
        self.declare_parameter('async_processing', True)
        self.declare_parameter('queue_size', 2)
        
        # Output parameters
        self.declare_parameter('publish_annotated_image', True)
        self.declare_parameter('publish_json', True)

        # Coral on-device parameters (only used when *_server_type == 'coral')
        self.declare_parameter('coral_model_path', '')
        self.declare_parameter('coral_labels_path', '')
        self.declare_parameter('coral_score_threshold', 0.4)
        self.declare_parameter('coral_model_name_tag', 'coral')
        # Per-camera color shift: left vs right use different palette rotations.
        self.declare_parameter('coral_palette_shift', 0)
        self.declare_parameter('coral_per_camera_palette', True)
        self.declare_parameter('coral_camera_palette_stride', 5)
        # Shown on annotated image header (empty => derived from model filename).
        self.declare_parameter('coral_model_display_name', '')

    def _get_parameters(self):
        """Get all parameter values."""
        self.primary_server_url = self.get_parameter('primary_server_url').value
        self.primary_server_type = self.get_parameter('primary_server_type').value
        self.fallback_server_url = self.get_parameter('fallback_server_url').value
        self.fallback_server_type = self.get_parameter('fallback_server_type').value
        self.api_key = self.get_parameter('api_key').value
        
        self.camera_topics = self.get_parameter('camera_topics').value
        self.use_compressed = self.get_parameter('use_compressed').value
        
        self.prompt = self.get_parameter('prompt').value
        self.confidence_threshold = self.get_parameter('confidence_threshold').value
        self.box_threshold = self.get_parameter('box_threshold').value
        self.text_threshold = self.get_parameter('text_threshold').value
        
        self.max_fps = self.get_parameter('max_fps').value
        self.resize_width = self.get_parameter('resize_width').value
        self.resize_height = self.get_parameter('resize_height').value
        self.jpeg_quality = self.get_parameter('jpeg_quality').value
        self.request_timeout = self.get_parameter('request_timeout').value
        self.async_processing = self.get_parameter('async_processing').value
        self.queue_size = self.get_parameter('queue_size').value
        
        self.publish_annotated = self.get_parameter('publish_annotated_image').value
        self.publish_json = self.get_parameter('publish_json').value

        self.coral_model_path = self.get_parameter('coral_model_path').value
        self.coral_labels_path = self.get_parameter('coral_labels_path').value
        self.coral_score_threshold = self.get_parameter('coral_score_threshold').value
        self.coral_model_name_tag = self.get_parameter('coral_model_name_tag').value
        self.coral_palette_shift = int(self.get_parameter('coral_palette_shift').value)
        self.coral_per_camera_palette = bool(self.get_parameter('coral_per_camera_palette').value)
        self.coral_camera_palette_stride = int(self.get_parameter('coral_camera_palette_stride').value)
        self.coral_model_display_name = (self.get_parameter('coral_model_display_name').value or '').strip()

        # Per-topic palette shift for annotated images (left ≠ right colors).
        npal = len(CLASS_COLOR_PALETTE)
        self._palette_shift_by_topic: Dict[str, int] = {}
        for i, t in enumerate(self.camera_topics):
            stride = self.coral_camera_palette_stride if self.coral_per_camera_palette else 0
            self._palette_shift_by_topic[t] = (
                self.coral_palette_shift + i * stride
            ) % npal

    def _init_clients(self):
        """Initialize AI server clients."""
        client_classes = {
            'grounding_dino': GroundingDINOClient,
            'yolo': YOLOClient,
            'sam': SAMClient,
            'coral': CoralClient,
        }

        def _make_client(name: str, server_type: str, server_url: str, priority: int):
            try:
                model_type = ModelType(server_type) if server_type != 'coral' else ModelType.CUSTOM
            except ValueError:
                model_type = ModelType.CUSTOM
            config = AIServerConfig(
                name=name,
                url=server_url or 'local://coral',
                model_type=model_type,
                api_key=self.api_key,
                timeout=self.request_timeout,
                priority=priority,
            )
            client_class = client_classes.get(server_type, GroundingDINOClient)

            if server_type == 'coral':
                return CoralClient(
                    config,
                    self.get_logger(),
                    model_path=self.coral_model_path,
                    labels_path=self.coral_labels_path,
                    score_threshold=self.coral_score_threshold,
                    model_name_tag=self.coral_model_name_tag,
                )
            return client_class(config, self.get_logger())

        if self.primary_server_url or self.primary_server_type == 'coral':
            try:
                self.ai_clients['primary'] = _make_client(
                    'primary', self.primary_server_type, self.primary_server_url, 0,
                )
            except Exception as e:
                self.get_logger().error(f"Failed to init primary client ({self.primary_server_type}): {e}")

        if self.fallback_server_url or self.fallback_server_type == 'coral':
            try:
                self.ai_clients['fallback'] = _make_client(
                    'fallback', self.fallback_server_type, self.fallback_server_url, 1,
                )
            except Exception as e:
                self.get_logger().warning(f"Failed to init fallback client ({self.fallback_server_type}): {e}")

        if 'primary' in self.ai_clients:
            self.active_client = self.ai_clients['primary']

    def _setup_ros_interfaces(self):
        """Setup ROS subscribers and publishers."""
        qos_profile = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            history=HistoryPolicy.KEEP_LAST,
            depth=1,
            durability=DurabilityPolicy.VOLATILE
        )

        self._topic_names: Dict[str, str] = {t: self._derive_camera_name(t) for t in self.camera_topics}
        self.last_detection_time_per_topic: Dict[str, float] = {t: 0.0 for t in self.camera_topics}

        self.image_subs = []
        for topic in self.camera_topics:
            if self.use_compressed:
                sub = self.create_subscription(
                    CompressedImage,
                    f'{topic}/compressed',
                    lambda msg, t=topic: self._compressed_callback(msg, t),
                    qos_profile
                )
            else:
                sub = self.create_subscription(
                    Image,
                    topic,
                    lambda msg, t=topic: self._image_callback(msg, t),
                    qos_profile
                )
            self.image_subs.append(sub)

        self.prompt_sub = self.create_subscription(
            String, '~/set_prompt', self._prompt_callback, 10
        )

        self.detection_pub = self.create_publisher(String, '~/detections', 10)
        self.status_pub = self.create_publisher(String, '~/status', 10)

        # Per-camera annotated image publishers, e.g. ~/left/annotated_image
        self.annotated_pubs: Dict[str, Any] = {}
        for topic, name in self._topic_names.items():
            self.annotated_pubs[topic] = self.create_publisher(Image, f'~/{name}/annotated_image', 10)

        # Single fallback publisher kept for backward-compat with existing tooling
        self.annotated_pub = self.create_publisher(Image, '~/annotated_image', 10)

    @staticmethod
    def _derive_camera_name(topic: str) -> str:
        """`/stereo/left/image_raw` -> `left`, `/camera/image_raw` -> `camera`."""
        parts = [p for p in topic.strip('/').split('/') if p]
        if not parts:
            return 'cam'
        if parts[-1] in ('image_raw', 'image_rect', 'image_rect_color', 'image', 'image_color') and len(parts) >= 2:
            return parts[-2]
        return parts[-1]

    def _prompt_callback(self, msg: String):
        """Update detection prompt."""
        new_prompt = msg.data.strip()
        if new_prompt:
            self.get_logger().info(f'Prompt updated: "{self.prompt}" -> "{new_prompt}"')
            self.prompt = new_prompt

    def _image_callback(self, msg: Image, topic: str):
        """Handle raw image."""
        self.stats['frames_received'] += 1
        
        if not self._should_process(topic):
            return
        
        try:
            cv_image = self.cv_bridge.imgmsg_to_cv2(msg, 'bgr8')
            self._queue_image(cv_image, msg.header, topic)
        except CvBridgeError as e:
            self.get_logger().error(f'CV Bridge error: {e}')

    def _compressed_callback(self, msg: CompressedImage, topic: str):
        """Handle compressed image."""
        self.stats['frames_received'] += 1
        
        if not self._should_process(topic):
            return
        
        try:
            np_arr = np.frombuffer(msg.data, np.uint8)
            cv_image = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
            if cv_image is not None:
                self._queue_image(cv_image, msg.header, topic)
        except Exception as e:
            self.get_logger().error(f'Decode error: {e}')

    def _should_process(self, topic: str) -> bool:
        """Per-camera throttle: each topic gets its own `max_fps` budget."""
        if not self.active_client or not self.active_client.is_healthy:
            return False

        current_time = time.time()
        last = self.last_detection_time_per_topic.get(topic, 0.0)
        if current_time - last < self.min_detection_interval:
            return False

        self.last_detection_time_per_topic[topic] = current_time
        self.last_detection_time = current_time
        return True

    def _queue_image(self, cv_image: np.ndarray, header: Header, topic: str):
        """Queue image for processing."""
        # On-device clients (Coral) handle resizing themselves and want full
        # original resolution so bounding boxes land in the right pixel space.
        is_local = isinstance(self.active_client, CoralClient)
        if not is_local and self.resize_width > 0 and self.resize_height > 0:
            cv_image = cv2.resize(
                cv_image,
                (self.resize_width, self.resize_height),
                interpolation=cv2.INTER_AREA
            )
        
        item = {
            'image': cv_image,
            'header': header,
            'topic': topic,
            'timestamp': time.time(),
        }
        
        if self.async_processing:
            try:
                self.processing_queue.put_nowait(item)
            except Exception:
                pass  # Queue full
        else:
            result = self._process_frame(item)
            if result:
                self._publish_results(result)

    def _processing_loop(self):
        """Async processing loop."""
        while not self.shutdown_flag:
            try:
                item = self.processing_queue.get(timeout=0.1)
                result = self._process_frame(item)
                if result:
                    self.result_queue.put(result)
            except Empty:
                continue
            except Exception as e:
                self.get_logger().error(f'Processing error: {e}')

    def _process_results(self):
        """Process results from queue."""
        try:
            while True:
                result = self.result_queue.get_nowait()
                self._publish_results(result)
        except Empty:
            pass

    def _process_frame(self, item: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Process single frame."""
        if not self.active_client:
            return None

        cv_image = item['image']

        detections = self.active_client.detect_image(
            cv_image,
            prompt=self.prompt,
            box_threshold=self.box_threshold,
            text_threshold=self.text_threshold,
            confidence=self.confidence_threshold,
            jpeg_quality=self.jpeg_quality,
        )
        
        if detections or self.active_client.is_healthy:
            self.stats['frames_processed'] += 1
            self.stats['detections_total'] += len(detections)
            
            # Update latency stats
            latency = self.active_client.last_latency_ms
            self.stats['latency_samples'].append(latency)
            if len(self.stats['latency_samples']) > 100:
                self.stats['latency_samples'].pop(0)
            self.stats['avg_latency_ms'] = sum(self.stats['latency_samples']) / len(self.stats['latency_samples'])
            
            return {
                'detections': [asdict(d) for d in detections],
                'header': {
                    'stamp_sec': item['header'].stamp.sec,
                    'stamp_nanosec': item['header'].stamp.nanosec,
                    'frame_id': item['header'].frame_id,
                },
                'source_topic': item['topic'],
                'prompt': self.prompt,
                'latency_ms': latency,
                'cv_image': cv_image,
            }
        
        return None

    def _publish_results(self, result: Dict[str, Any]):
        """Publish detection results."""
        if self.publish_json:
            json_result = {k: v for k, v in result.items() if k != 'cv_image'}
            msg = String()
            msg.data = json.dumps(json_result, default=_json_default)
            self.detection_pub.publish(msg)

        if self.publish_annotated and 'cv_image' in result:
            src = result.get('source_topic', '')
            palette_shift = self._palette_shift_by_topic.get(src, self.coral_palette_shift)
            cam_name = self._topic_names.get(src, src)
            lat = result.get('latency_ms', 0.0)
            n = len(result['detections'])
            fps_txt = f'{1000.0 / max(lat, 0.001):.0f} FPS' if lat > 0 else ''
            annotated = draw_detection_overlay(
                result['cv_image'].copy(),
                result['detections'],
                palette_shift=palette_shift,
                latency_ms=0.0,
                header_top=f'{cam_name}  |  {self._annotate_model_title}  |  palette+{palette_shift}',
                header_sub=f'detections:{n}  |  {lat:.1f} ms  |  {fps_txt}',
            )
            try:
                img_msg = self.cv_bridge.cv2_to_imgmsg(annotated, 'bgr8')
                src = result.get('source_topic', '')
                # Preserve incoming frame_id so RViz can place it correctly.
                hdr = result.get('header', {})
                img_msg.header.stamp = self.get_clock().now().to_msg()
                img_msg.header.frame_id = hdr.get('frame_id', '')
                pub = self.annotated_pubs.get(src, self.annotated_pub)
                pub.publish(img_msg)
            except CvBridgeError:
                pass

    def _health_check_all(self):
        """Check health of all servers."""
        for name, client in self.ai_clients.items():
            is_healthy = client.health_check()
            self.get_logger().debug(f'{name} health: {is_healthy}')
        
        # Select best available client
        for name in ['primary', 'fallback']:
            if name in self.ai_clients and self.ai_clients[name].is_healthy:
                if self.active_client != self.ai_clients[name]:
                    self.get_logger().info(f'Switching to {name} server')
                self.active_client = self.ai_clients[name]
                break

    def _publish_stats(self):
        """Publish node statistics."""
        status = {
            'timestamp': datetime.now().isoformat(),
            'active_server': self.active_client.config.name if self.active_client else None,
            'prompt': self.prompt,
            'camera_topics': self.camera_topics,
            'stats': self.stats.copy(),
        }
        status['stats'].pop('latency_samples', None)
        
        msg = String()
        msg.data = json.dumps(status, default=_json_default)
        self.status_pub.publish(msg)

    def _log_startup(self):
        """Log startup information."""
        self.get_logger().info('='*60)
        self.get_logger().info('AI Vision Node initialized')
        self.get_logger().info(f'  Primary Server: {self.primary_server_url}')
        self.get_logger().info(f'  Model Type: {self.primary_server_type}')
        self.get_logger().info(f'  Camera Topics: {self.camera_topics}')
        self.get_logger().info(f'  Detection Prompt: {self.prompt}')
        self.get_logger().info('='*60)

    def destroy_node(self):
        """Cleanup."""
        self.shutdown_flag = True
        for client in self.ai_clients.values():
            try:
                client.session.close()
            except Exception:
                pass
        try:
            super().destroy_node()
        except Exception:
            pass


def main(args=None):
    rclpy.init(args=args)
    node = AIVisionNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()

