# Status message for AI server health monitoring

std_msgs/Header header

# Server identification
string server_name
string server_url
string model_name

# Connection status
bool is_connected
bool is_healthy
uint32 consecutive_failures
string last_error

# Performance metrics
float32 avg_latency_ms
float32 min_latency_ms
float32 max_latency_ms
uint32 requests_total
uint32 requests_success
uint32 requests_failed

# Server capabilities
string[] supported_models
string[] supported_prompts

