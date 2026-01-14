#!/usr/bin/env python3
"""
Launch file for Multi-Model AI Vision on ROS2 camera topics.

Supports multiple AI backends:
- Grounding DINO (open-vocabulary detection)
- YOLO (real-time object detection)
- SAM (Segment Anything Model)

Usage:
    ros2 launch deepgis_vision ai_vision.launch.py

    # With YOLO backend:
    ros2 launch deepgis_vision ai_vision.launch.py \
        primary_server_type:=yolo

    # With multiple cameras:
    ros2 launch deepgis_vision ai_vision.launch.py \
        camera_topics:="['/stereo/left/image_raw', '/stereo/right/image_raw']"
"""

import os
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, GroupAction
from launch.substitutions import LaunchConfiguration
from launch.conditions import IfCondition
from launch_ros.actions import Node, PushRosNamespace
from ament_index_python.packages import get_package_share_directory


def generate_launch_description():
    pkg_share = get_package_share_directory('deepgis_vision')
    config_file = os.path.join(pkg_share, 'config', 'ai_vision.yaml')
    
    # Declare launch arguments
    args = [
        DeclareLaunchArgument(
            'primary_server_url',
            default_value='http://192.168.0.232:5000',
            description='URL of the primary AI server'
        ),
        DeclareLaunchArgument(
            'primary_server_type',
            default_value='grounding_dino',
            description='Type of primary AI model (grounding_dino, yolo, sam)'
        ),
        DeclareLaunchArgument(
            'fallback_server_url',
            default_value='',
            description='URL of the fallback AI server (optional)'
        ),
        DeclareLaunchArgument(
            'camera_topics',
            default_value="['/stereo/right/image_raw']",
            description='List of camera topics to subscribe'
        ),
        DeclareLaunchArgument(
            'prompt',
            default_value='person . car . tree . building',
            description='Detection prompt for open-vocabulary models'
        ),
        DeclareLaunchArgument(
            'max_fps',
            default_value='0.2',
            description='Maximum detection rate'
        ),
        DeclareLaunchArgument(
            'namespace',
            default_value='',
            description='Namespace for all nodes'
        ),
        DeclareLaunchArgument(
            'enable_visualizer',
            default_value='true',
            description='Enable detection visualizer node'
        ),
    ]
    
    # AI Vision node
    ai_vision_node = Node(
        package='deepgis_vision',
        executable='ai_vision_node.py',
        name='ai_vision_node',
        output='screen',
        parameters=[
            config_file,
            {
                'primary_server_url': LaunchConfiguration('primary_server_url'),
                'primary_server_type': LaunchConfiguration('primary_server_type'),
                'fallback_server_url': LaunchConfiguration('fallback_server_url'),
                'prompt': LaunchConfiguration('prompt'),
                'max_fps': LaunchConfiguration('max_fps'),
            }
        ],
    )
    
    # Visualizer node
    visualizer_node = Node(
        package='deepgis_vision',
        executable='detection_visualizer.py',
        name='detection_visualizer',
        output='screen',
        condition=IfCondition(LaunchConfiguration('enable_visualizer')),
        parameters=[{
            'detection_topic': '/ai_vision_node/detections',
            'show_confidence': True,
            'show_fps': True,
            'show_latency': True,
        }]
    )
    
    return LaunchDescription(args + [ai_vision_node, visualizer_node])

