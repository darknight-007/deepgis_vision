# Array of detections for a single image frame
# Published by AI vision nodes after processing

std_msgs/Header header

# Source image information
uint32 image_width
uint32 image_height
string image_encoding
string source_topic

# AI Server information
string server_url
string model_name
string model_version

# Detection parameters used
string prompt               # Text prompt for open-vocabulary detection
float32 confidence_threshold
float32 nms_threshold

# Processing metrics
float32 total_inference_time_ms
float32 preprocessing_time_ms
float32 postprocessing_time_ms
float32 network_latency_ms

# Detections
deepgis_vision/Detection[] detections

