#!/usr/bin/env python3
"""
On-device Coral Edge TPU perception for the earth-rover stereo cameras.

Subscribes to:
    /stereo/left/image_raw
    /stereo/right/image_raw

Publishes (per camera):
    /ai_vision_node/left/annotated_image    sensor_msgs/Image with bboxes + label + score
    /ai_vision_node/right/annotated_image
    /ai_vision_node/detections              std_msgs/String JSON, includes source_topic
    /ai_vision_node/status                  std_msgs/String JSON, node stats

Default model: SSDLite MobileDet COCO (90 classes), fetched during Coral setup
into ~/Downloads/coral/samples. Override with launch args:

    ros2 launch deepgis_vision coral_perception.launch.py \\
        coral_model_path:=/path/to/your_edgetpu.tflite \\
        coral_labels_path:=/path/to/labels.txt

To open the debug image streams:
    ros2 run rqt_image_view rqt_image_view /ai_vision_node/left/annotated_image
    ros2 run rqt_image_view rqt_image_view /ai_vision_node/right/annotated_image
"""

import os
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, OpaqueFunction
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node
from ament_index_python.packages import get_package_share_directory


_HOME = os.path.expanduser('~')
_DEFAULT_MODEL = os.path.join(
    _HOME, 'Downloads', 'coral', 'samples',
    'ssdlite_mobiledet_coco_qat_postprocess_edgetpu.tflite',
)
_DEFAULT_LABELS = os.path.join(
    _HOME, 'Downloads', 'coral', 'samples', 'coco_labels.txt',
)


def _launch_setup(context, *args, **kwargs):
    pkg_share = get_package_share_directory('deepgis_vision')
    config_file = os.path.join(pkg_share, 'config', 'coral_perception.yaml')

    def cfg(name: str) -> str:
        return LaunchConfiguration(name).perform(context)

    use_compressed_str = cfg('use_compressed').strip().lower()
    use_compressed = use_compressed_str in ('true', '1', 'yes', 'y')

    params = {
        'camera_topics': [cfg('left_topic'), cfg('right_topic')],
        'coral_model_path': cfg('coral_model_path'),
        'coral_labels_path': cfg('coral_labels_path'),
        'coral_score_threshold': float(cfg('coral_score_threshold')),
        'coral_model_name_tag': cfg('coral_model_name_tag'),
        'coral_model_display_name': cfg('coral_model_display_name'),
        'coral_palette_shift': int(float(cfg('coral_palette_shift'))),
        'coral_per_camera_palette': cfg(
            'coral_per_camera_palette').strip().lower() in ('true', '1', 'yes'),
        'coral_camera_palette_stride': int(float(cfg('coral_camera_palette_stride'))),
        'max_fps': float(cfg('max_fps')),
        'use_compressed': use_compressed,
    }

    return [
        Node(
            package='deepgis_vision',
            executable='ai_vision_node.py',
            name='ai_vision_node',
            output='screen',
            parameters=[config_file, params],
        ),
    ]


def generate_launch_description():
    args = [
        DeclareLaunchArgument(
            'left_topic',
            default_value='/stereo/left/image_raw',
            description='Left camera image_raw topic.',
        ),
        DeclareLaunchArgument(
            'right_topic',
            default_value='/stereo/right/image_raw',
            description='Right camera image_raw topic.',
        ),
        DeclareLaunchArgument(
            'coral_model_path',
            default_value=_DEFAULT_MODEL,
            description='Absolute path to a *_edgetpu.tflite detection model.',
        ),
        DeclareLaunchArgument(
            'coral_labels_path',
            default_value=_DEFAULT_LABELS,
            description='Absolute path to the labels.txt file.',
        ),
        DeclareLaunchArgument(
            'coral_score_threshold',
            default_value='0.4',
            description='Minimum confidence to emit a detection.',
        ),
        DeclareLaunchArgument(
            'coral_model_name_tag',
            default_value='MobileDet',
            description='Stored in Detection JSON (`model_name` field).',
        ),
        DeclareLaunchArgument(
            'coral_model_display_name',
            default_value='',
            description='Header on annotated streams; blank = basename of model file.',
        ),
        DeclareLaunchArgument(
            'coral_palette_shift',
            default_value='0',
            description='Base palette rotation (per-class BGR hues).',
        ),
        DeclareLaunchArgument(
            'coral_per_camera_palette',
            default_value='true',
            description='If true, each camera topic increments palette separately.',
        ),
        DeclareLaunchArgument(
            'coral_camera_palette_stride',
            default_value='5',
            description='Palette index increment between successive camera_topics.',
        ),
        DeclareLaunchArgument(
            'max_fps',
            default_value='15.0',
            description='Per-camera maximum detection rate.',
        ),
        DeclareLaunchArgument(
            'use_compressed',
            default_value='false',
            description='Subscribe to <topic>/compressed instead of raw.',
        ),
    ]

    return LaunchDescription(args + [OpaqueFunction(function=_launch_setup)])
