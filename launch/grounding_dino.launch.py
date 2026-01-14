#!/usr/bin/env python3
"""
Launch file for Grounding DINO object detection on ROS2 camera topics.

This launch file starts the Grounding DINO vision node that:
- Subscribes to camera topics (e.g., Grasshopper right camera)
- Sends images to a remote Grounding DINO AI server
- Publishes detection results and annotated images

Usage:
    ros2 launch deepgis_vision grounding_dino.launch.py

    # With custom server URL:
    ros2 launch deepgis_vision grounding_dino.launch.py \
        ai_server_url:=http://192.168.0.232:5000

    # With custom camera topic:
    ros2 launch deepgis_vision grounding_dino.launch.py \
        camera_topic:=/stereo/left/image_raw

    # With custom detection prompt:
    ros2 launch deepgis_vision grounding_dino.launch.py \
        prompt:="dog . cat . bird"
"""

import os
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution
from launch_ros.actions import Node
from launch_ros.substitutions import FindPackageShare
from ament_index_python.packages import get_package_share_directory


def generate_launch_description():
    # Get package share directory
    pkg_share = get_package_share_directory('deepgis_vision')
    
    # Declare launch arguments
    ai_server_url_arg = DeclareLaunchArgument(
        'ai_server_url',
        default_value='http://192.168.0.232:5000',
        description='URL of the Grounding DINO AI server'
    )
    
    camera_topic_arg = DeclareLaunchArgument(
        'camera_topic',
        default_value='/stereo/right/image_raw',
        description='Camera topic to subscribe to'
    )
    
    prompt_arg = DeclareLaunchArgument(
        'prompt',
        default_value='person . vehicle . tree . building',
        description='Detection prompt (classes separated by " . ")'
    )
    
    max_fps_arg = DeclareLaunchArgument(
        'max_fps',
        default_value='0.2',
        description='Maximum detection rate (FPS)'
    )
    
    confidence_threshold_arg = DeclareLaunchArgument(
        'confidence_threshold',
        default_value='0.3',
        description='Minimum confidence threshold for detections'
    )
    
    use_config_arg = DeclareLaunchArgument(
        'use_config',
        default_value='true',
        description='Whether to use YAML config file'
    )
    
    # Config file path
    config_file = os.path.join(pkg_share, 'config', 'grounding_dino.yaml')
    
    # Grounding DINO node
    grounding_dino_node = Node(
        package='deepgis_vision',
        executable='grounding_dino_node.py',
        name='grounding_dino_node',
        output='screen',
        parameters=[
            config_file,
            {
                'ai_server_url': LaunchConfiguration('ai_server_url'),
                'camera_topic': LaunchConfiguration('camera_topic'),
                'prompt': LaunchConfiguration('prompt'),
                'max_fps': LaunchConfiguration('max_fps'),
                'confidence_threshold': LaunchConfiguration('confidence_threshold'),
            }
        ],
        remappings=[
            # Remap topics if needed
            # ('~/detections', '/detections'),
        ]
    )
    
    # Detection visualizer node (optional)
    visualizer_node = Node(
        package='deepgis_vision',
        executable='detection_visualizer.py',
        name='detection_visualizer',
        output='screen',
        parameters=[{
            'image_topic': LaunchConfiguration('camera_topic'),
            'detection_topic': '/grounding_dino_node/detections',
            'show_confidence': True,
            'show_fps': True,
            'show_latency': True,
        }]
    )
    
    return LaunchDescription([
        # Launch arguments
        ai_server_url_arg,
        camera_topic_arg,
        prompt_arg,
        max_fps_arg,
        confidence_threshold_arg,
        use_config_arg,
        
        # Nodes
        grounding_dino_node,
        visualizer_node,
    ])

