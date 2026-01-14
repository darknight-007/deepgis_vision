#!/usr/bin/env python3
"""
Launch file for Grasshopper3 stereo camera AI vision processing.

This launch configuration:
1. Optionally starts the Grasshopper3 stereo cameras
2. Runs AI detection on the right camera
3. Provides visualization of detection results

Usage:
    # Just vision processing (cameras already running):
    ros2 launch deepgis_vision grasshopper_stereo_vision.launch.py

    # With camera launch:
    ros2 launch deepgis_vision grasshopper_stereo_vision.launch.py \
        start_cameras:=true

    # Custom AI server:
    ros2 launch deepgis_vision grasshopper_stereo_vision.launch.py \
        ai_server_url:=http://192.168.0.232:5000
"""

import os
from launch import LaunchDescription
from launch.actions import (
    DeclareLaunchArgument,
    IncludeLaunchDescription,
    GroupAction,
)
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution
from launch.conditions import IfCondition
from launch_ros.actions import Node
from ament_index_python.packages import get_package_share_directory


def generate_launch_description():
    # Package directories
    deepgis_vision_share = get_package_share_directory('deepgis_vision')
    
    # Try to get spinnaker camera driver share (may not be installed)
    try:
        spinnaker_share = get_package_share_directory('spinnaker_camera_driver')
        has_spinnaker = True
    except Exception:
        spinnaker_share = None
        has_spinnaker = False
    
    # Launch arguments
    args = [
        DeclareLaunchArgument(
            'start_cameras',
            default_value='false',
            description='Whether to start the Grasshopper cameras'
        ),
        DeclareLaunchArgument(
            'ai_server_url',
            default_value='http://192.168.0.232:5000',
            description='URL of the AI server (Grounding DINO)'
        ),
        DeclareLaunchArgument(
            'camera_topic',
            default_value='/stereo/right/image_raw',
            description='Camera topic for detection'
        ),
        DeclareLaunchArgument(
            'prompt',
            default_value='person . vehicle . tree . building . road sign',
            description='Detection prompt'
        ),
        DeclareLaunchArgument(
            'max_fps',
            default_value='0.2',
            description='Maximum detection FPS'
        ),
        DeclareLaunchArgument(
            'enable_visualizer',
            default_value='true',
            description='Enable visualization node'
        ),
    ]
    
    # Config file
    config_file = os.path.join(deepgis_vision_share, 'config', 'grounding_dino.yaml')
    
    nodes = []
    
    # Optionally include camera launch
    if has_spinnaker:
        camera_launch = IncludeLaunchDescription(
            PythonLaunchDescriptionSource(
                os.path.join(spinnaker_share, 'launch', 'grasshopper_stereo.launch.py')
            ),
            condition=IfCondition(LaunchConfiguration('start_cameras'))
        )
        nodes.append(camera_launch)
    
    # Grounding DINO detection node
    detection_node = Node(
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
            }
        ]
    )
    nodes.append(detection_node)
    
    # Visualizer node
    visualizer_node = Node(
        package='deepgis_vision',
        executable='detection_visualizer.py',
        name='detection_visualizer',
        output='screen',
        condition=IfCondition(LaunchConfiguration('enable_visualizer')),
        parameters=[{
            'image_topic': LaunchConfiguration('camera_topic'),
            'detection_topic': '/grounding_dino_node/detections',
        }]
    )
    nodes.append(visualizer_node)
    
    return LaunchDescription(args + nodes)

