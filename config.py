"""
Configuration file for Car Parking Detection System
All configurable parameters are centralized here for easy management
"""

from typing import Final, List, Tuple

# Backend URL
BACKEND_URL: Final[str] = "http://localhost:8000"

# MJPEG stream server (AI service serves annotated frames here)
MJPEG_STREAM_PORT: Final[int] = 8081
AI_STREAM_URL: Final[str] = f"http://localhost:{MJPEG_STREAM_PORT}/stream"

# Parking Space Dimensions
PARKING_WIDTH: Final[int] = 107
PARKING_HEIGHT: Final[int] = 48

# Detection Thresholds
OCCUPANCY_THRESHOLD: Final[int] = 2000  # Pixel count threshold for space occupancy
CONFIDENCE_THRESHOLD: Final[float] = 0.5  # Minimum confidence for vehicle detection

# Temporal Smoothing
# A slot only changes its displayed state when this fraction of recent frames
# agree.  Higher values = more stable but slower to react.  Start with 5.
TEMPORAL_SMOOTH_FRAMES: Final[int] = 3

# Gate / Entry-point Configuration
# 0-indexed position of the gate slot in posList (slot 11 in the UI = index 10).
# Set this to match whichever slot number you labelled as the gate during --setup.
GATE_SLOT_INDEX: Final[int] = 10
EXIT_GATE_SLOT_INDEX: Final[int] = 11

# Real-world scale (optional).  Set to the number of pixels that represent 1 metre
# in your camera view so the path-guidance banner can show an approximate distance.
# Leave at 0 to disable the metre label.
PIXELS_PER_METER: Final[float] = 0.0

# Parking event database (SQLite)
PARKING_DB_FILE: Final[str] = "data/parking_events.db"

# How many video frames to show the "Parked in Slot N!" confirmation banner
# before returning to IDLE state.  At ~30 fps, 90 ≈ 3 seconds.
PARKED_CONFIRM_FRAMES: Final[int] = 90

# YOLO Model Configuration
MODEL_PATH: Final[str] = "yolov8n.pt"
CAR_CLASSES: Final[List[int]] = [2, 3, 5, 7]  # COCO classes: cars, motorcycles, buses, trucks

# File Paths
POSITION_FILE: Final[str] = "CarParkPos"
BOUNDARY_FILE: Final[str] = "data/CarParkBoundary.pkl"
DEFAULT_VIDEO_PATH: Final[str] = "carPark.mp4"
DEFAULT_IMAGE_PATH: Final[str] = "carParkImg.jpg"

# Camera / Live-stream Configuration
# Set CAMERA_URL to the RTSP / HTTP stream URL of your IP camera.
# Examples:
#   RTSP  : "rtsp://user:pass@192.168.1.100:554/stream1"
#   HTTP  : "http://192.168.1.100:8080/video"
#   MJPEG : "http://192.168.1.100/mjpg/video.mjpg"
CAMERA_URL: Final[str] = ""  # leave empty; override via --camera CLI flag
CAMERA_RECONNECT_DELAY: Final[int] = 3  # seconds to wait before reconnecting on failure
CAMERA_FRAME_TIMEOUT: Final[int] = 30  # frames without a read before reconnecting

# Supabase Configuration (optional, for cloud sync)
# Example:
# SUPABASE_URL = "https://<project-ref>.supabase.co"
# SUPABASE_KEY = "<anon-or-service-role-key>"
SUPABASE_URL: Final[str] = "https://ysjpxserjwbthaownacy.supabase.co"
SUPABASE_KEY: Final[str] = (
    "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpc3MiOiJzdXBhYmFzZSIsInJlZiI6InlzanB4c2VyandidGhhb3duYWN5Iiwicm9sZSI6ImFub24iLCJpYXQiOjE3NzAwNzcwMzIsImV4cCI6MjA4NTY1MzAzMn0.Y2wAa7nii7UnNtezsWPx_1yVv43lRdsGdtPnBHRG6Y0"
)
SUPABASE_LOT_ID: Final[str] = "9cfae083-b963-40be-bb4a-03c1d1aa4cd8"
SUPABASE_SLOTS_TABLE: Final[str] = "parking_slots"
SUPABASE_SESSIONS_TABLE: Final[str] = "parking_sessions"

# Directory Structure
REPORTS_DIR: Final[str] = "reports"
DATA_DIR: Final[str] = "data"
MODELS_DIR: Final[str] = "models"

# CSV Configuration
CSV_FILE: Final[str] = "parking_status.csv"
CSV_APPEND_MODE: Final[bool] = True  # Set to True to keep historical data

# UI Configuration
BORDER_SIZE: Final[int] = 50
WINDOW_NAME: Final[str] = "Parking Detection"

# Zoom Configuration
MIN_ZOOM: Final[float] = 0.5
MAX_ZOOM: Final[float] = 3.0
ZOOM_STEP: Final[float] = 0.1
PAN_STEP: Final[int] = 50

# Colors (BGR format)
COLOR_EMPTY: Final[Tuple[int, int, int]] = (0, 255, 0)  # Green
COLOR_OCCUPIED: Final[Tuple[int, int, int]] = (0, 0, 255)  # Red
COLOR_SPECIAL: Final[Tuple[int, int, int]] = (0, 255, 255)  # Yellow
COLOR_REGULAR: Final[Tuple[int, int, int]] = (255, 0, 255)  # Magenta
COLOR_SELECTION: Final[Tuple[int, int, int]] = (0, 255, 0)  # Green

# Special Parking Detection
YELLOW_PERCENTAGE_THRESHOLD: Final[int] = (
    3  # Minimum percentage of yellow pixels for special parking
)
MIN_LINE_LENGTH_RATIO: Final[float] = 0.3  # Minimum line length as ratio of space dimension

# Report Configuration
REPORT_DPI: Final[int] = 300
REPORT_FIGSIZE: Final[Tuple[int, int]] = (15, 10)
