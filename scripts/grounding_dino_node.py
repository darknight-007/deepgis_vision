#!/usr/bin/env python3
"""
Grounding DINO Vision Node for ROS2

Subscribes to camera image topics and performs open-vocabulary object detection
using a remote Grounding DINO AI server. Based on the DeepGIS-XR AI server
architecture pattern.

Supports:
- Real-time detection on camera streams
- Dynamic prompt updates
- Multiple AI server backends
- Health monitoring and failover
"""

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy, DurabilityPolicy

from sensor_msgs.msg import Image, CompressedImage
from std_msgs.msg import String, Header
from geometry_msgs.msg import Point

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
import io


class GroundingDINONode(Node):
    """
    ROS2 Node for Grounding DINO object detection via remote AI server.
    
    Based on the DeepGIS-XR remote AI architecture:
    https://github.com/Earth-Innovation-Hub/deepgis-xr
    """

    def __init__(self):
        super().__init__('grounding_dino_node')

        # Declare parameters
        self.declare_parameter('ai_server_url', 'http://localhost:5000')
        self.declare_parameter('detection_endpoint', '/api/predict')
        self.declare_parameter('health_endpoint', '/health')
        self.declare_parameter('api_key', '')
        
        # Camera parameters
        self.declare_parameter('camera_topic', '/stereo/right/image_raw')
        self.declare_parameter('use_compressed', False)
        self.declare_parameter('image_transport', 'raw')
        
        # Detection parameters
        self.declare_parameter('prompt', 'person . car . tree . building')
        self.declare_parameter('confidence_threshold', 0.3)
        self.declare_parameter('nms_threshold', 0.5)
        self.declare_parameter('box_threshold', 0.35)
        self.declare_parameter('text_threshold', 0.25)
        
        # Performance parameters
        self.declare_parameter('max_fps', 10.0)  # Limit detection rate
        self.declare_parameter('resize_width', 640)  # Resize before sending
        self.declare_parameter('resize_height', 480)
        self.declare_parameter('jpeg_quality', 85)  # Compression quality
        self.declare_parameter('request_timeout', 5.0)  # API timeout
        self.declare_parameter('async_processing', True)  # Use async queue
        self.declare_parameter('queue_size', 2)  # Max pending requests
        
        # Health monitoring
        self.declare_parameter('health_check_interval', 10.0)
        self.declare_parameter('max_consecutive_failures', 5)
        self.declare_parameter('retry_delay', 2.0)
        
        # Output parameters
        self.declare_parameter('publish_annotated_image', True)
        self.declare_parameter('publish_detections', True)
        
        # Get parameters
        self.server_url = self.get_parameter('ai_server_url').value
        self.detection_endpoint = self.get_parameter('detection_endpoint').value
        self.health_endpoint = self.get_parameter('health_endpoint').value
        self.api_key = self.get_parameter('api_key').value
        
        self.camera_topic = self.get_parameter('camera_topic').value
        self.use_compressed = self.get_parameter('use_compressed').value
        
        self.prompt = self.get_parameter('prompt').value
        self.confidence_threshold = self.get_parameter('confidence_threshold').value
        self.nms_threshold = self.get_parameter('nms_threshold').value
        self.box_threshold = self.get_parameter('box_threshold').value
        self.text_threshold = self.get_parameter('text_threshold').value
        
        self.max_fps = self.get_parameter('max_fps').value
        self.resize_width = self.get_parameter('resize_width').value
        self.resize_height = self.get_parameter('resize_height').value
        self.jpeg_quality = self.get_parameter('jpeg_quality').value
        self.request_timeout = self.get_parameter('request_timeout').value
        self.async_processing = self.get_parameter('async_processing').value
        self.queue_size = self.get_parameter('queue_size').value
        
        self.health_check_interval = self.get_parameter('health_check_interval').value
        self.max_consecutive_failures = self.get_parameter('max_consecutive_failures').value
        
        self.publish_annotated = self.get_parameter('publish_annotated_image').value
        self.publish_detections = self.get_parameter('publish_detections').value

        # API endpoints
        self.api_endpoints = {
            'detect': f'{self.server_url}{self.detection_endpoint}',
            'health': f'{self.server_url}{self.health_endpoint}',
        }

        # State management
        self.cv_bridge = CvBridge()
        self.data_lock = Lock()
        self.is_connected = False
        self.consecutive_failures = 0
        self.last_detection_time = 0.0
        self.min_detection_interval = 1.0 / self.max_fps
        
        # Statistics
        self.stats = {
            'frames_received': 0,
            'frames_processed': 0,
            'detections_total': 0,
            'avg_latency_ms': 0.0,
            'min_latency_ms': float('inf'),
            'max_latency_ms': 0.0,
            'total_latency_ms': 0.0,
        }
        
        # Async processing queue
        self.processing_queue: Queue = Queue(maxsize=self.queue_size)
        self.result_queue: Queue = Queue()
        self.processing_thread: Optional[Thread] = None
        self.shutdown_flag = False

        # HTTP session with connection pooling
        self.http_session = requests.Session()
        if self.api_key:
            self.http_session.headers.update({'Authorization': f'Bearer {self.api_key}'})

        # QoS profile for camera topics
        qos_profile = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            history=HistoryPolicy.KEEP_LAST,
            depth=1,
            durability=DurabilityPolicy.VOLATILE
        )

        # Subscribers
        if self.use_compressed:
            self.image_sub = self.create_subscription(
                CompressedImage,
                f'{self.camera_topic}/compressed',
                self.compressed_image_callback,
                qos_profile
            )
        else:
            self.image_sub = self.create_subscription(
                Image,
                self.camera_topic,
                self.image_callback,
                qos_profile
            )

        # Prompt update subscriber
        self.prompt_sub = self.create_subscription(
            String,
            '~/set_prompt',
            self.prompt_callback,
            10
        )

        # Publishers
        self.detection_pub = self.create_publisher(
            String,  # JSON detections (use custom msg after build)
            '~/detections',
            10
        )
        
        self.annotated_pub = self.create_publisher(
            Image,
            '~/annotated_image',
            10
        )
        
        self.status_pub = self.create_publisher(
            String,  # JSON status
            '~/status',
            10
        )

        # Health check timer
        self.health_timer = self.create_timer(
            self.health_check_interval,
            self.health_check
        )
        
        # Stats publishing timer
        self.stats_timer = self.create_timer(5.0, self.publish_stats)

        # Start async processing thread
        if self.async_processing:
            self.processing_thread = Thread(target=self._processing_loop, daemon=True)
            self.processing_thread.start()
            
            # Result processing timer
            self.result_timer = self.create_timer(0.01, self._process_results)

        # Initial health check
        self.health_check()

        self.get_logger().info('='*60)
        self.get_logger().info('Grounding DINO Vision Node initialized')
        self.get_logger().info(f'  AI Server URL: {self.server_url}')
        self.get_logger().info(f'  Camera Topic: {self.camera_topic}')
        self.get_logger().info(f'  Detection Prompt: {self.prompt}')
        self.get_logger().info(f'  Max FPS: {self.max_fps}')
        self.get_logger().info(f'  Async Processing: {self.async_processing}')
        self.get_logger().info('='*60)

    def health_check(self):
        """Check AI server health status."""
        try:
            response = self.http_session.get(
                self.api_endpoints['health'],
                timeout=3.0
            )
            if response.status_code == 200:
                self.is_connected = True
                self.consecutive_failures = 0
                self.get_logger().debug('AI server health check: OK')
            else:
                self._handle_connection_failure(f'Health check returned {response.status_code}')
        except requests.exceptions.RequestException as e:
            self._handle_connection_failure(str(e))

    def _handle_connection_failure(self, error_msg: str):
        """Handle connection failures with backoff."""
        self.consecutive_failures += 1
        if self.consecutive_failures >= self.max_consecutive_failures:
            if self.is_connected:
                self.get_logger().error(
                    f'AI server disconnected after {self.consecutive_failures} failures: {error_msg}'
                )
            self.is_connected = False
        else:
            self.get_logger().warning(
                f'AI server health check failed ({self.consecutive_failures}/'
                f'{self.max_consecutive_failures}): {error_msg}'
            )

    def prompt_callback(self, msg: String):
        """Update detection prompt dynamically."""
        new_prompt = msg.data.strip()
        if new_prompt:
            old_prompt = self.prompt
            self.prompt = new_prompt
            self.get_logger().info(f'Prompt updated: "{old_prompt}" -> "{new_prompt}"')

    def image_callback(self, msg: Image):
        """Process incoming raw image."""
        self.stats['frames_received'] += 1
        
        # Rate limiting
        current_time = time.time()
        if current_time - self.last_detection_time < self.min_detection_interval:
            return
            
        if not self.is_connected:
            return

        try:
            cv_image = self.cv_bridge.imgmsg_to_cv2(msg, 'bgr8')
            self._process_image(cv_image, msg.header)
        except CvBridgeError as e:
            self.get_logger().error(f'CV Bridge error: {e}')

    def compressed_image_callback(self, msg: CompressedImage):
        """Process incoming compressed image."""
        self.stats['frames_received'] += 1
        
        # Rate limiting
        current_time = time.time()
        if current_time - self.last_detection_time < self.min_detection_interval:
            return
            
        if not self.is_connected:
            return

        try:
            np_arr = np.frombuffer(msg.data, np.uint8)
            cv_image = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
            if cv_image is not None:
                self._process_image(cv_image, msg.header)
        except Exception as e:
            self.get_logger().error(f'Compressed image decode error: {e}')

    def _process_image(self, cv_image: np.ndarray, header: Header):
        """Queue or process image for detection."""
        self.last_detection_time = time.time()
        
        # Resize for faster transmission
        original_size = (cv_image.shape[1], cv_image.shape[0])
        if self.resize_width > 0 and self.resize_height > 0:
            cv_image = cv2.resize(
                cv_image,
                (self.resize_width, self.resize_height),
                interpolation=cv2.INTER_AREA
            )
        
        if self.async_processing:
            # Queue for async processing
            try:
                self.processing_queue.put_nowait({
                    'image': cv_image,
                    'header': header,
                    'original_size': original_size,
                    'timestamp': time.time(),
                })
            except Exception:
                # Queue full, skip this frame
                pass
        else:
            # Synchronous processing
            result = self._detect_objects(cv_image, header, original_size)
            if result:
                self._publish_results(result)

    def _processing_loop(self):
        """Async processing thread loop."""
        while not self.shutdown_flag:
            try:
                item = self.processing_queue.get(timeout=0.1)
                result = self._detect_objects(
                    item['image'],
                    item['header'],
                    item['original_size']
                )
                if result:
                    self.result_queue.put(result)
            except Empty:
                continue
            except Exception as e:
                self.get_logger().error(f'Processing error: {e}')

    def _process_results(self):
        """Process results from async queue (called by timer)."""
        try:
            while True:
                result = self.result_queue.get_nowait()
                self._publish_results(result)
        except Empty:
            pass

    def _detect_objects(
        self,
        cv_image: np.ndarray,
        header: Header,
        original_size: Tuple[int, int]
    ) -> Optional[Dict[str, Any]]:
        """
        Send image to Grounding DINO AI server for detection.
        
        API Format (multipart/form-data):
        - file: <image binary>
        - text_prompt: "person . car . tree"
        - box_threshold: 0.35
        - text_threshold: 0.25
        
        Response Format:
        {
            "predictions": {
                "count": 2,
                "phrases": ["person", "car"],
                "logits": [0.89, 0.75],
                "boxes": [[x1, y1, x2, y2], ...]
            },
            "annotated_image": <base64 JPEG>
        }
        """
        start_time = time.time()
        
        try:
            # Encode image to JPEG bytes
            encode_params = [cv2.IMWRITE_JPEG_QUALITY, self.jpeg_quality]
            _, buffer = cv2.imencode('.jpg', cv_image, encode_params)
            image_bytes = buffer.tobytes()
            
            # Prepare multipart form data
            files = {
                'file': ('image.jpg', image_bytes, 'image/jpeg')
            }
            data = {
                'text_prompt': self.prompt,
                'box_threshold': str(self.box_threshold),
                'text_threshold': str(self.text_threshold),
            }
            
            # Send request
            response = self.http_session.post(
                self.api_endpoints['detect'],
                files=files,
                data=data,
                timeout=self.request_timeout
            )
            
            if response.status_code == 200:
                result = response.json()
                
                # Calculate latency
                latency_ms = (time.time() - start_time) * 1000
                self._update_latency_stats(latency_ms)
                
                # Convert predictions format to internal format
                predictions = result.get('predictions', {})
                detections = []
                
                if predictions.get('count', 0) > 0:
                    phrases = predictions.get('phrases', [])
                    logits = predictions.get('logits', [])
                    boxes = predictions.get('boxes', [])
                    
                    for i in range(len(phrases)):
                        detections.append({
                            'label': phrases[i],
                            'confidence': logits[i] if i < len(logits) else 0.0,
                            'box': boxes[i] if i < len(boxes) else [0, 0, 1, 1]
                        })
                
                # Build result in original format for compatibility
                result_compat = {
                    'detections': detections,
                    'header': {
                        'stamp_sec': header.stamp.sec,
                        'stamp_nanosec': header.stamp.nanosec,
                        'frame_id': header.frame_id,
                    },
                    'original_size': original_size,
                    'processed_size': (cv_image.shape[1], cv_image.shape[0]),
                    'latency_ms': latency_ms,
                    'prompt': self.prompt,
                    'source_topic': self.camera_topic,
                    'cv_image': cv_image,
                }
                
                self.consecutive_failures = 0
                self.stats['frames_processed'] += 1
                
                return result_compat
            else:
                self._handle_connection_failure(f'Detection returned {response.status_code}')
                return None
                
        except requests.exceptions.Timeout:
            self._handle_connection_failure('Request timeout')
            return None
        except requests.exceptions.RequestException as e:
            self._handle_connection_failure(str(e))
            return None
        except Exception as e:
            self.get_logger().error(f'Detection error: {e}')
            return None

    def _update_latency_stats(self, latency_ms: float):
        """Update latency statistics."""
        with self.data_lock:
            self.stats['total_latency_ms'] += latency_ms
            self.stats['min_latency_ms'] = min(self.stats['min_latency_ms'], latency_ms)
            self.stats['max_latency_ms'] = max(self.stats['max_latency_ms'], latency_ms)
            if self.stats['frames_processed'] > 0:
                self.stats['avg_latency_ms'] = (
                    self.stats['total_latency_ms'] / self.stats['frames_processed']
                )

    def _publish_results(self, result: Dict[str, Any]):
        """Publish detection results."""
        detections = result.get('detections', [])
        self.stats['detections_total'] += len(detections)
        
        # Publish detection JSON
        if self.publish_detections:
            detection_msg = String()
            # Remove cv_image from JSON output
            json_result = {k: v for k, v in result.items() if k != 'cv_image'}
            detection_msg.data = json.dumps(json_result)
            self.detection_pub.publish(detection_msg)
        
        # Publish annotated image
        if self.publish_annotated and 'cv_image' in result:
            annotated = self._draw_detections(
                result['cv_image'].copy(),
                detections,
                result.get('original_size', (640, 480))
            )
            try:
                img_msg = self.cv_bridge.cv2_to_imgmsg(annotated, 'bgr8')
                img_msg.header.stamp = self.get_clock().now().to_msg()
                img_msg.header.frame_id = result['header'].get('frame_id', 'camera')
                self.annotated_pub.publish(img_msg)
            except CvBridgeError as e:
                self.get_logger().error(f'CV Bridge error: {e}')

    def _draw_detections(
        self,
        image: np.ndarray,
        detections: List[Dict],
        original_size: Tuple[int, int]
    ) -> np.ndarray:
        """Draw bounding boxes and labels on image."""
        h, w = image.shape[:2]
        
        # Color palette for different classes
        colors = [
            (0, 255, 0),    # Green
            (255, 0, 0),    # Blue
            (0, 0, 255),    # Red
            (255, 255, 0),  # Cyan
            (255, 0, 255),  # Magenta
            (0, 255, 255),  # Yellow
            (128, 0, 255),  # Purple
            (255, 128, 0),  # Orange
        ]
        
        class_colors = {}
        
        for det in detections:
            label = det.get('label', 'unknown')
            confidence = det.get('confidence', 0.0)
            box = det.get('box', [0, 0, 1, 1])
            
            # Assign color to class
            if label not in class_colors:
                class_colors[label] = colors[len(class_colors) % len(colors)]
            color = class_colors[label]
            
            # Convert normalized coordinates to pixel coordinates
            x1 = int(box[0] * w)
            y1 = int(box[1] * h)
            x2 = int(box[2] * w)
            y2 = int(box[3] * h)
            
            # Draw bounding box
            cv2.rectangle(image, (x1, y1), (x2, y2), color, 2)
            
            # Draw label background
            text = f'{label}: {confidence:.2f}'
            (text_w, text_h), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
            cv2.rectangle(image, (x1, y1 - text_h - 8), (x1 + text_w + 4, y1), color, -1)
            
            # Draw label text
            cv2.putText(
                image, text, (x1 + 2, y1 - 4),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1
            )
        
        # Draw info overlay
        info_text = f'Detections: {len(detections)} | Prompt: {self.prompt[:30]}...'
        cv2.putText(image, info_text, (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        
        return image

    def publish_stats(self):
        """Publish node statistics."""
        status = {
            'timestamp': datetime.now().isoformat(),
            'server_url': self.server_url,
            'is_connected': self.is_connected,
            'consecutive_failures': self.consecutive_failures,
            'prompt': self.prompt,
            'camera_topic': self.camera_topic,
            'stats': {
                'frames_received': self.stats['frames_received'],
                'frames_processed': self.stats['frames_processed'],
                'detections_total': self.stats['detections_total'],
                'avg_latency_ms': round(self.stats['avg_latency_ms'], 2),
                'min_latency_ms': round(self.stats['min_latency_ms'], 2) 
                    if self.stats['min_latency_ms'] != float('inf') else 0.0,
                'max_latency_ms': round(self.stats['max_latency_ms'], 2),
            }
        }
        
        msg = String()
        msg.data = json.dumps(status)
        self.status_pub.publish(msg)

    def destroy_node(self):
        """Cleanup before node shutdown."""
        self.shutdown_flag = True
        
        # Wait for processing thread
        if self.processing_thread and self.processing_thread.is_alive():
            self.processing_thread.join(timeout=2.0)
        
        # Close HTTP session
        try:
            self.http_session.close()
        except Exception:
            pass
        
        try:
            super().destroy_node()
        except Exception:
            pass


def main(args=None):
    rclpy.init(args=args)
    
    node = GroundingDINONode()
    
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()

