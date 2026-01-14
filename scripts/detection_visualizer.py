#!/usr/bin/env python3
"""
Detection Visualizer Node for ROS2

Subscribes to detection results and visualizes them on images.
Supports multiple visualization modes and overlays.
"""

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy, DurabilityPolicy

from sensor_msgs.msg import Image
from std_msgs.msg import String

import cv2
import numpy as np
from cv_bridge import CvBridge, CvBridgeError
import json
from typing import Dict, List, Any
from collections import deque
from datetime import datetime


class DetectionVisualizer(Node):
    """
    Visualization node for AI detection results.
    
    Features:
    - Bounding box overlays with labels
    - Detection history tracking
    - Performance metrics display
    - Multiple color schemes
    """

    def __init__(self):
        super().__init__('detection_visualizer')

        # Parameters
        self.declare_parameter('image_topic', '/stereo/right/image_raw')
        self.declare_parameter('detection_topic', '/grounding_dino_node/detections')
        self.declare_parameter('show_confidence', True)
        self.declare_parameter('show_fps', True)
        self.declare_parameter('show_latency', True)
        self.declare_parameter('box_thickness', 2)
        self.declare_parameter('font_scale', 0.5)
        self.declare_parameter('history_length', 30)  # Frames to average FPS
        
        self.image_topic = self.get_parameter('image_topic').value
        self.detection_topic = self.get_parameter('detection_topic').value
        self.show_confidence = self.get_parameter('show_confidence').value
        self.show_fps = self.get_parameter('show_fps').value
        self.show_latency = self.get_parameter('show_latency').value
        self.box_thickness = self.get_parameter('box_thickness').value
        self.font_scale = self.get_parameter('font_scale').value
        self.history_length = self.get_parameter('history_length').value

        # State
        self.cv_bridge = CvBridge()
        self.latest_image = None
        self.latest_detections = []
        self.latest_metadata = {}
        self.frame_times = deque(maxlen=self.history_length)
        self.last_frame_time = None
        
        # Color palette (BGR)
        self.colors = [
            (0, 255, 0),    # Green
            (255, 128, 0),  # Blue-ish
            (0, 0, 255),    # Red
            (255, 255, 0),  # Cyan
            (255, 0, 255),  # Magenta
            (0, 255, 255),  # Yellow
            (128, 0, 255),  # Purple
            (0, 128, 255),  # Orange
            (255, 0, 128),  # Pink
            (128, 255, 0),  # Lime
        ]
        self.class_colors: Dict[str, tuple] = {}

        # QoS
        qos = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            history=HistoryPolicy.KEEP_LAST,
            depth=1,
            durability=DurabilityPolicy.VOLATILE
        )

        # Subscribers
        self.image_sub = self.create_subscription(
            Image, self.image_topic, self.image_callback, qos
        )
        self.detection_sub = self.create_subscription(
            String, self.detection_topic, self.detection_callback, 10
        )

        # Publisher
        self.viz_pub = self.create_publisher(Image, '~/visualization', 10)
        
        # Visualization timer (30 FPS)
        self.viz_timer = self.create_timer(1.0 / 30.0, self.visualize)

        self.get_logger().info(f'Detection Visualizer initialized')
        self.get_logger().info(f'  Image Topic: {self.image_topic}')
        self.get_logger().info(f'  Detection Topic: {self.detection_topic}')

    def image_callback(self, msg: Image):
        """Store latest image."""
        try:
            self.latest_image = self.cv_bridge.imgmsg_to_cv2(msg, 'bgr8')
            
            # FPS calculation
            current_time = self.get_clock().now()
            if self.last_frame_time:
                dt = (current_time - self.last_frame_time).nanoseconds / 1e9
                if dt > 0:
                    self.frame_times.append(1.0 / dt)
            self.last_frame_time = current_time
            
        except CvBridgeError as e:
            self.get_logger().error(f'CV Bridge error: {e}')

    def detection_callback(self, msg: String):
        """Store latest detections."""
        try:
            data = json.loads(msg.data)
            self.latest_detections = data.get('detections', [])
            self.latest_metadata = {
                'latency_ms': data.get('latency_ms', 0),
                'prompt': data.get('prompt', ''),
                'source_topic': data.get('source_topic', ''),
            }
        except json.JSONDecodeError as e:
            self.get_logger().error(f'JSON decode error: {e}')

    def get_class_color(self, class_name: str) -> tuple:
        """Get consistent color for class."""
        if class_name not in self.class_colors:
            idx = len(self.class_colors) % len(self.colors)
            self.class_colors[class_name] = self.colors[idx]
        return self.class_colors[class_name]

    def visualize(self):
        """Create and publish visualization."""
        if self.latest_image is None:
            return
        
        image = self.latest_image.copy()
        h, w = image.shape[:2]
        
        # Draw detections
        for det in self.latest_detections:
            self.draw_detection(image, det, w, h)
        
        # Draw info overlay
        self.draw_overlay(image)
        
        # Publish
        try:
            msg = self.cv_bridge.cv2_to_imgmsg(image, 'bgr8')
            msg.header.stamp = self.get_clock().now().to_msg()
            self.viz_pub.publish(msg)
        except CvBridgeError as e:
            self.get_logger().error(f'CV Bridge error: {e}')

    def draw_detection(self, image: np.ndarray, det: Dict, w: int, h: int):
        """Draw single detection on image."""
        # Get class info
        class_name = det.get('class_name', det.get('label', 'unknown'))
        confidence = det.get('confidence', 0.0)
        color = self.get_class_color(class_name)
        
        # Calculate bounding box
        if 'bbox_x_norm' in det:
            x1 = int(det['bbox_x_norm'] * w)
            y1 = int(det['bbox_y_norm'] * h)
            x2 = int((det['bbox_x_norm'] + det['bbox_width_norm']) * w)
            y2 = int((det['bbox_y_norm'] + det['bbox_height_norm']) * h)
        elif 'box' in det:
            box = det['box']
            x1 = int(box[0] * w)
            y1 = int(box[1] * h)
            x2 = int(box[2] * w)
            y2 = int(box[3] * h)
        else:
            return
        
        # Draw box
        cv2.rectangle(image, (x1, y1), (x2, y2), color, self.box_thickness)
        
        # Draw label
        if self.show_confidence:
            label = f'{class_name}: {confidence:.2f}'
        else:
            label = class_name
        
        (tw, th), baseline = cv2.getTextSize(
            label, cv2.FONT_HERSHEY_SIMPLEX, self.font_scale, 1
        )
        
        # Label background
        cv2.rectangle(
            image,
            (x1, y1 - th - 10),
            (x1 + tw + 8, y1),
            color, -1
        )
        
        # Label text
        cv2.putText(
            image, label,
            (x1 + 4, y1 - 5),
            cv2.FONT_HERSHEY_SIMPLEX,
            self.font_scale,
            (255, 255, 255),
            1, cv2.LINE_AA
        )
        
        # Center dot
        cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
        cv2.circle(image, (cx, cy), 4, color, -1)

    def draw_overlay(self, image: np.ndarray):
        """Draw info overlay."""
        h, w = image.shape[:2]
        y_offset = 25
        line_height = 22
        
        # Semi-transparent background
        overlay = image.copy()
        cv2.rectangle(overlay, (5, 5), (300, 5 + line_height * 4), (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.5, image, 0.5, 0, image)
        
        # Detection count
        text = f'Detections: {len(self.latest_detections)}'
        cv2.putText(image, text, (10, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        y_offset += line_height
        
        # FPS
        if self.show_fps and self.frame_times:
            fps = sum(self.frame_times) / len(self.frame_times)
            text = f'FPS: {fps:.1f}'
            cv2.putText(image, text, (10, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
            y_offset += line_height
        
        # Latency
        if self.show_latency:
            latency = self.latest_metadata.get('latency_ms', 0)
            text = f'Latency: {latency:.0f}ms'
            color = (0, 255, 0) if latency < 100 else (0, 165, 255) if latency < 200 else (0, 0, 255)
            cv2.putText(image, text, (10, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
            y_offset += line_height
        
        # Prompt
        prompt = self.latest_metadata.get('prompt', '')
        if prompt:
            text = f'Prompt: {prompt[:30]}...' if len(prompt) > 30 else f'Prompt: {prompt}'
            cv2.putText(image, text, (10, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)

    def destroy_node(self):
        """Cleanup."""
        try:
            super().destroy_node()
        except Exception:
            pass


def main(args=None):
    rclpy.init(args=args)
    node = DetectionVisualizer()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()

