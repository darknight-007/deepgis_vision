#!/usr/bin/env python3
"""
Launch file for Grasshopper3 stereo camera AI vision processing.

This launch configuration:
1. Optionally starts the Grasshopper3 stereo cameras
2. Runs AI detection on left and/or right camera
3. Provides visualization of detection results

Usage:
    # Both cameras (default):
    ros2 launch deepgis_vision grasshopper_stereo_vision.launch.py

    # With camera launch:
    ros2 launch deepgis_vision grasshopper_stereo_vision.launch.py \
        start_cameras:=true

    # Right camera only:
    ros2 launch deepgis_vision grasshopper_stereo_vision.launch.py \
        enable_left:=false

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
            'enable_left',
            default_value='true',
            description='Enable detection on left camera'
        ),
        DeclareLaunchArgument(
            'enable_right',
            default_value='true',
            description='Enable detection on right camera'
        ),
        DeclareLaunchArgument(
            'prompt',
            default_value='person . vehicle . tree . building . road sign',
            description='Detection prompt'
        ),
        DeclareLaunchArgument(
            'max_fps',
            default_value='0.2',
            description='Maximum detection FPS per camera'
        ),
        DeclareLaunchArgument(
            'enable_visualizer',
            default_value='true',
            description='Enable visualization nodes'
        ),
    ]
    
    # Config file
    config_file = os.path.join(deepgis_vision_share, 'config', 'grounding_dino.yaml')
    
    nodes = []
    
    # Optionally include camera launch
    if has_spinnaker:
        camera_launch = IncludeLaunchDescription(
            PythonLaunchDescriptionSource(
                os.path.join(spinnaker_share, 'launch', 'grasshopper_stereo_min.launch.py')
            ),
            condition=IfCondition(LaunchConfiguration('start_cameras'))
        )
        nodes.append(camera_launch)
    
    # Left camera detection node
    left_detection_node = Node(
        package='deepgis_vision',
        executable='grounding_dino_node.py',
        name='grounding_dino_left',
        namespace='stereo/left',
        output='screen',
        condition=IfCondition(LaunchConfiguration('enable_left')),
        parameters=[
            config_file,
            {
                'ai_server_url': LaunchConfiguration('ai_server_url'),
                'camera_topic': '/stereo/left/image_raw',
                'prompt': LaunchConfiguration('prompt'),
                'max_fps': LaunchConfiguration('max_fps'),
            }
        ]
    )
    nodes.append(left_detection_node)
    
    # Right camera detection node
    right_detection_node = Node(
        package='deepgis_vision',
        executable='grounding_dino_node.py',
        name='grounding_dino_right',
        namespace='stereo/right',
        output='screen',
        condition=IfCondition(LaunchConfiguration('enable_right')),
        parameters=[
            config_file,
            {
                'ai_server_url': LaunchConfiguration('ai_server_url'),
                'camera_topic': '/stereo/right/image_raw',
                'prompt': LaunchConfiguration('prompt'),
                'max_fps': LaunchConfiguration('max_fps'),
            }
        ]
    )
    nodes.append(right_detection_node)
    
    # Left camera visualizer
    left_visualizer_node = Node(
        package='deepgis_vision',
        executable='detection_visualizer.py',
        name='detection_visualizer_left',
        namespace='stereo/left',
        output='screen',
        condition=IfCondition(LaunchConfiguration('enable_left')),
        parameters=[{
            'image_topic': '/stereo/left/image_raw',
            'detection_topic': '/stereo/left/grounding_dino_left/detections',
        }]
    )
    nodes.append(left_visualizer_node)
    
    # Right camera visualizer
    right_visualizer_node = Node(
        package='deepgis_vision',
        executable='detection_visualizer.py',
        name='detection_visualizer_right',
        namespace='stereo/right',
        output='screen',
        condition=IfCondition(LaunchConfiguration('enable_right')),
        parameters=[{
            'image_topic': '/stereo/right/image_raw',
            'detection_topic': '/stereo/right/grounding_dino_right/detections',
        }]
    )
    nodes.append(right_visualizer_node)
    
    return LaunchDescription(args + nodes)

