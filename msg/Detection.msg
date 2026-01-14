# Single detection result from AI vision model
# Compatible with Grounding DINO, YOLO, and other object detectors

std_msgs/Header header

# Detection ID (unique per frame)
uint32 id

# Class information
string class_name           # Detected class/label (e.g., "person", "car")
uint32 class_id             # Class ID if using predefined classes
float32 confidence          # Detection confidence [0.0, 1.0]

# Bounding box in image coordinates (pixels)
float32 bbox_x              # Top-left X
float32 bbox_y              # Top-left Y
float32 bbox_width          # Width
float32 bbox_height         # Height

# Normalized bounding box [0.0, 1.0]
float32 bbox_x_norm
float32 bbox_y_norm
float32 bbox_width_norm
float32 bbox_height_norm

# Center point
float32 center_x
float32 center_y

# Additional metadata
string prompt               # Text prompt used (for open-vocabulary detection)
string model_name           # Model that generated this detection
float32 inference_time_ms   # Time taken for this detection

# Optional segmentation mask (RLE encoded or empty)
uint8[] mask_rle
uint32 mask_width
uint32 mask_height

