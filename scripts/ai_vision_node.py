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
        self.clients: Dict[str, AIServerClient] = {}
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

    def _init_clients(self):
        """Initialize AI server clients."""
        client_classes = {
            'grounding_dino': GroundingDINOClient,
            'yolo': YOLOClient,
            'sam': SAMClient,
        }
        
        # Primary server
        if self.primary_server_url:
            config = AIServerConfig(
                name='primary',
                url=self.primary_server_url,
                model_type=ModelType(self.primary_server_type),
                api_key=self.api_key,
                timeout=self.request_timeout,
                priority=0,
            )
            client_class = client_classes.get(self.primary_server_type, GroundingDINOClient)
            self.clients['primary'] = client_class(config, self.get_logger())
        
        # Fallback server
        if self.fallback_server_url:
            config = AIServerConfig(
                name='fallback',
                url=self.fallback_server_url,
                model_type=ModelType(self.fallback_server_type),
                api_key=self.api_key,
                timeout=self.request_timeout,
                priority=1,
            )
            client_class = client_classes.get(self.fallback_server_type, YOLOClient)
            self.clients['fallback'] = client_class(config, self.get_logger())
        
        # Set active client
        if 'primary' in self.clients:
            self.active_client = self.clients['primary']

    def _setup_ros_interfaces(self):
        """Setup ROS subscribers and publishers."""
        qos_profile = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            history=HistoryPolicy.KEEP_LAST,
            depth=1,
            durability=DurabilityPolicy.VOLATILE
        )
        
        # Subscribe to camera topics
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
        
        # Dynamic prompt update
        self.prompt_sub = self.create_subscription(
            String, '~/set_prompt', self._prompt_callback, 10
        )
        
        # Publishers
        self.detection_pub = self.create_publisher(String, '~/detections', 10)
        self.annotated_pub = self.create_publisher(Image, '~/annotated_image', 10)
        self.status_pub = self.create_publisher(String, '~/status', 10)

    def _prompt_callback(self, msg: String):
        """Update detection prompt."""
        new_prompt = msg.data.strip()
        if new_prompt:
            self.get_logger().info(f'Prompt updated: "{self.prompt}" -> "{new_prompt}"')
            self.prompt = new_prompt

    def _image_callback(self, msg: Image, topic: str):
        """Handle raw image."""
        self.stats['frames_received'] += 1
        
        if not self._should_process():
            return
        
        try:
            cv_image = self.cv_bridge.imgmsg_to_cv2(msg, 'bgr8')
            self._queue_image(cv_image, msg.header, topic)
        except CvBridgeError as e:
            self.get_logger().error(f'CV Bridge error: {e}')

    def _compressed_callback(self, msg: CompressedImage, topic: str):
        """Handle compressed image."""
        self.stats['frames_received'] += 1
        
        if not self._should_process():
            return
        
        try:
            np_arr = np.frombuffer(msg.data, np.uint8)
            cv_image = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
            if cv_image is not None:
                self._queue_image(cv_image, msg.header, topic)
        except Exception as e:
            self.get_logger().error(f'Decode error: {e}')

    def _should_process(self) -> bool:
        """Check if we should process this frame."""
        if not self.active_client or not self.active_client.is_healthy:
            return False
        
        current_time = time.time()
        if current_time - self.last_detection_time < self.min_detection_interval:
            return False
        
        self.last_detection_time = current_time
        return True

    def _queue_image(self, cv_image: np.ndarray, header: Header, topic: str):
        """Queue image for processing."""
        # Resize
        if self.resize_width > 0 and self.resize_height > 0:
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
        
        # Encode image
        encode_params = [cv2.IMWRITE_JPEG_QUALITY, self.jpeg_quality]
        _, buffer = cv2.imencode('.jpg', cv_image, encode_params)
        image_base64 = base64.b64encode(buffer).decode('utf-8')
        
        # Detect
        detections = self.active_client.detect(
            image_base64,
            prompt=self.prompt,
            box_threshold=self.box_threshold,
            text_threshold=self.text_threshold,
            confidence=self.confidence_threshold,
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
        # JSON detections
        if self.publish_json:
            json_result = {k: v for k, v in result.items() if k != 'cv_image'}
            msg = String()
            msg.data = json.dumps(json_result)
            self.detection_pub.publish(msg)
        
        # Annotated image
        if self.publish_annotated and 'cv_image' in result:
            annotated = self._draw_detections(result['cv_image'].copy(), result['detections'])
            try:
                img_msg = self.cv_bridge.cv2_to_imgmsg(annotated, 'bgr8')
                img_msg.header.stamp = self.get_clock().now().to_msg()
                self.annotated_pub.publish(img_msg)
            except CvBridgeError:
                pass

    def _draw_detections(self, image: np.ndarray, detections: List[Dict]) -> np.ndarray:
        """Draw detections on image."""
        h, w = image.shape[:2]
        colors = [(0, 255, 0), (255, 0, 0), (0, 0, 255), (255, 255, 0), (255, 0, 255)]
        class_colors = {}
        
        for det in detections:
            label = det.get('class_name', 'unknown')
            conf = det.get('confidence', 0.0)
            
            if label not in class_colors:
                class_colors[label] = colors[len(class_colors) % len(colors)]
            color = class_colors[label]
            
            # Calculate pixel coordinates
            x1 = int(det.get('bbox_x_norm', 0) * w)
            y1 = int(det.get('bbox_y_norm', 0) * h)
            x2 = int((det.get('bbox_x_norm', 0) + det.get('bbox_width_norm', 0)) * w)
            y2 = int((det.get('bbox_y_norm', 0) + det.get('bbox_height_norm', 0)) * h)
            
            cv2.rectangle(image, (x1, y1), (x2, y2), color, 2)
            
            text = f'{label}: {conf:.2f}'
            (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
            cv2.rectangle(image, (x1, y1 - th - 8), (x1 + tw + 4, y1), color, -1)
            cv2.putText(image, text, (x1 + 2, y1 - 4), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        
        info = f'Detections: {len(detections)} | {self.prompt[:25]}...'
        cv2.putText(image, info, (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        
        return image

    def _health_check_all(self):
        """Check health of all servers."""
        for name, client in self.clients.items():
            is_healthy = client.health_check()
            self.get_logger().debug(f'{name} health: {is_healthy}')
        
        # Select best available client
        for name in ['primary', 'fallback']:
            if name in self.clients and self.clients[name].is_healthy:
                if self.active_client != self.clients[name]:
                    self.get_logger().info(f'Switching to {name} server')
                self.active_client = self.clients[name]
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
        msg.data = json.dumps(status)
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
        for client in self.clients.values():
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

