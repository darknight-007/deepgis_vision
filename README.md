# DeepGIS Vision - ROS2 AI Vision Package

A ROS2 package for real-time computer vision operations using third-party AI servers. Designed for integration with the [DeepGIS-XR](https://github.com/Earth-Innovation-Hub/deepgis-xr) AI infrastructure.

## Features

- **Multi-Model Support**: Grounding DINO, YOLO, SAM, and custom models
- **Open-Vocabulary Detection**: Detect any object using natural language prompts
- **Remote AI Server Architecture**: Offload computation to GPU servers
- **Automatic Failover**: Switch between primary and fallback servers
- **Real-time Processing**: Async processing with rate limiting
- **Grasshopper Camera Integration**: Pre-configured for FLIR Grasshopper3 stereo cameras

## Supported AI Models

| Model | Type | Description |
|-------|------|-------------|
| Grounding DINO | Open-Vocabulary Detection | Detect objects using text prompts |
| YOLOv8 | Real-time Detection | Fast object detection with predefined classes |
| SAM | Segmentation | Segment Anything Model for instance segmentation |

## Installation

### Prerequisites

- ROS2 Humble or later
- Python 3.8+
- OpenCV
- cv_bridge

### Build

```bash
cd ~/ros2_ws
colcon build --packages-select deepgis_vision
source install/setup.bash
```

## Quick Start

### 1. Start the AI Server

The package expects a Grounding DINO server running. You can use the DeepGIS-XR Docker setup:

```bash
# Using DeepGIS-XR docker-compose
docker-compose up -d

# Or run a standalone Grounding DINO server
# Server should expose: POST /detect/grounding_dino
```

### 2. Launch Detection

```bash
# Basic launch with Grasshopper right camera
ros2 launch deepgis_vision grounding_dino.launch.py

# With custom AI server
ros2 launch deepgis_vision grounding_dino.launch.py \
    ai_server_url:=http://192.168.0.232:5000

# With custom detection prompt
ros2 launch deepgis_vision grounding_dino.launch.py \
    prompt:="dog . cat . bird . person"

# With custom camera topic
ros2 launch deepgis_vision grounding_dino.launch.py \
    camera_topic:=/camera/color/image_raw
```

### 3. View Results

```bash
# View detection results (JSON)
ros2 topic echo /grounding_dino_node/detections

# View annotated images
ros2 run rqt_image_view rqt_image_view /grounding_dino_node/annotated_image

# View visualization with overlay
ros2 run rqt_image_view rqt_image_view /detection_visualizer/visualization
```

## Nodes

### `grounding_dino_node`

Main detection node for Grounding DINO open-vocabulary object detection.

#### Subscribed Topics

| Topic | Type | Description |
|-------|------|-------------|
| `<camera_topic>` | `sensor_msgs/Image` | Raw camera image |
| `~/set_prompt` | `std_msgs/String` | Dynamic prompt update |

#### Published Topics

| Topic | Type | Description |
|-------|------|-------------|
| `~/detections` | `std_msgs/String` | JSON detection results |
| `~/annotated_image` | `sensor_msgs/Image` | Image with bounding boxes |
| `~/status` | `std_msgs/String` | Node status and statistics |

#### Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `ai_server_url` | string | `http://localhost:5000` | AI server URL |
| `camera_topic` | string | `/stereo/right/image_raw` | Camera topic |
| `prompt` | string | `person . car . tree` | Detection prompt |
| `confidence_threshold` | float | `0.3` | Minimum confidence |
| `max_fps` | float | `10.0` | Maximum detection rate |
| `resize_width` | int | `640` | Resize width before sending |
| `resize_height` | int | `480` | Resize height before sending |
| `async_processing` | bool | `true` | Use async queue |

### `ai_vision_node`

Multi-model AI vision node with automatic failover.

#### Additional Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `primary_server_url` | string | `http://localhost:5000` | Primary server URL |
| `primary_server_type` | string | `grounding_dino` | Model type |
| `fallback_server_url` | string | `` | Fallback server URL |
| `fallback_server_type` | string | `yolo` | Fallback model type |
| `camera_topics` | string[] | `[/stereo/right/image_raw]` | Multiple camera topics |

### `detection_visualizer`

Visualization node for detection overlay.

## AI Server API

The package expects an AI server with the following API format (DeepGIS-XR compatible):

### Grounding DINO Endpoint

**POST** `/detect/grounding_dino`

Request:
```json
{
    "image": "<base64 encoded JPEG>",
    "prompt": "person . car . tree",
    "box_threshold": 0.35,
    "text_threshold": 0.25
}
```

Response:
```json
{
    "detections": [
        {
            "label": "person",
            "confidence": 0.89,
            "box": [0.1, 0.2, 0.5, 0.8]
        }
    ],
    "inference_time": 0.123
}
```

### Health Endpoint

**GET** `/health`

Response: HTTP 200 if healthy

## Configuration

### YAML Configuration

Configuration files are located in `config/`:

- `grounding_dino.yaml` - Grounding DINO specific settings
- `ai_vision.yaml` - Multi-model configuration

### Example: Custom Server Configuration

```yaml
grounding_dino_node:
  ros__parameters:
    ai_server_url: "http://192.168.0.232:5000"
    camera_topic: "/stereo/right/image_raw"
    prompt: "person . vehicle . obstacle"
    max_fps: 15.0
    confidence_threshold: 0.4
```

## Launch Files

| Launch File | Description |
|-------------|-------------|
| `grounding_dino.launch.py` | Grounding DINO detection |
| `ai_vision.launch.py` | Multi-model detection |
| `grasshopper_stereo_vision.launch.py` | Full stereo camera + detection |

## Dynamic Prompt Update

Update detection prompt at runtime:

```bash
# Using ros2 topic
ros2 topic pub --once /grounding_dino_node/set_prompt std_msgs/String "data: 'car . truck . bus'"

# Using service (if available)
ros2 service call /grounding_dino_node/set_prompt deepgis_vision/srv/SetPrompt "{prompt: 'dog . cat'}"
```

## Performance Tuning

### Reduce Latency

```yaml
max_fps: 5.0          # Lower FPS = less load
resize_width: 480     # Smaller images
resize_height: 360
jpeg_quality: 75      # More compression
async_processing: true
```

### Maximize Throughput

```yaml
max_fps: 30.0         # Higher FPS
resize_width: 0       # No resize (full resolution)
resize_height: 0
queue_size: 4         # Larger queue
```

## Message Types

### Detection.msg

```
std_msgs/Header header
uint32 id
string class_name
float32 confidence
float32 bbox_x_norm
float32 bbox_y_norm
float32 bbox_width_norm
float32 bbox_height_norm
string prompt
string model_name
```

### DetectionArray.msg

```
std_msgs/Header header
uint32 image_width
uint32 image_height
string source_topic
string model_name
float32 total_inference_time_ms
deepgis_vision/Detection[] detections
```

## Integration with Grasshopper3 Cameras

The package is pre-configured for FLIR Grasshopper3 stereo cameras:

- **Left Camera**: Serial 22312692, Topic `/stereo/left/image_raw`
- **Right Camera**: Serial 22312674, Topic `/stereo/right/image_raw`

```bash
# Start cameras (if using spinnaker_camera_driver)
ros2 launch spinnaker_camera_driver grasshopper_stereo_min.launch.py \
    left_serial:=22312692 \
    right_serial:=22312674 \
    parameter_file:=grasshopper.yaml

# Start AI detection on right camera
ros2 launch deepgis_vision grounding_dino.launch.py \
    camera_topic:=/stereo/right/image_raw
```

## Troubleshooting

### AI Server Connection Issues

```bash
# Check server health
curl http://localhost:5000/health

# View node status
ros2 topic echo /grounding_dino_node/status
```

### No Detections

1. Check camera topic is publishing: `ros2 topic hz /stereo/right/image_raw`
2. Verify prompt format: `"class1 . class2 . class3"`
3. Lower confidence threshold: `confidence_threshold:=0.2`

### High Latency

1. Reduce image size: `resize_width:=320 resize_height:=240`
2. Lower FPS: `max_fps:=5.0`
3. Check network connection to AI server

## References

- [DeepGIS-XR](https://github.com/Earth-Innovation-Hub/deepgis-xr) - AI server architecture
- [Grounding DINO](https://github.com/IDEA-Research/GroundingDINO) - Open-vocabulary detection
- [FLIR Spinnaker ROS2 Driver](https://github.com/ros-drivers/flir_camera_driver)

## License

MIT License - See LICENSE file for details.

## Contributing

Contributions welcome! Please follow the coding standards in the main project.

