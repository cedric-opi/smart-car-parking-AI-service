# 🚗 Advanced Car Parking Space Detection System
**An intelligent parking space detection system powered by computer vision and deep learning**

## 🎓 What Does This Do? (In Simple Terms)

Imagine you're managing a parking lot and need to know:
- ✅ How many parking spaces are empty right now?
- ✅ Which specific spots are available?
- ✅ How full is the parking lot over time?

**This system does exactly that - automatically!**

You provide an image or video of your parking lot, and the system:
1. Lets you mark where the parking spaces are (just click and drag)
2. Uses AI to detect cars in those spaces
3. Shows you which spaces are empty (green) and occupied (red)
4. Generates detailed reports with charts and statistics

**No manual counting needed!** The computer does all the work.

### Real-World Example

```
Input:  Picture, Video or even Live Camera of Parking Lot.
Output: "35 spaces occupied, 15 available"
        + Visual map showing which spots are free
        + Detailed report
        + CSV file for data analysis
```

---

This project is **excellent for academic purposes** because it demonstrates:

**Computer Vision Concepts:**
- Image processing (grayscale, blurring, thresholding)
- Object detection using deep learning (YOLOv8)
- Real-time video processing
- Color space transformations (RGB to HSV)
- Morphological operations

**Software Engineering Skills:**
- Clean code architecture
- Configuration management
- Error handling and logging
- File I/O operations
- CLI application development

**Data Science & Analytics:**
- Data collection and storage (CSV)
- Statistical analysis
- Data visualization (charts, graphs)
- Report generation

### Citations & Academic Use

```
Bharath K. (2024). Advanced Car Parking Space Detection System.
GitHub repository. https://github.com/8harath/Car-Parking-Detection
```

---

## 🌟 Features

### 1. **Interactive Parking Space Selection**
- 🖱️ Drag-and-select multiple parking spaces at once
- 📐 Grid-based automatic space allocation
- 👁️ Visual feedback during selection
- ↩️ Undo/Redo functionality
- 🗑️ Right-click to remove individual spaces
- 🔍 Zoom and pan capabilities

### 2. **Intelligent Vehicle Detection**
- 🤖 YOLOv8-based deep learning model
- 🚙 Detects: Cars, Motorcycles, Buses, Trucks
- 📊 Confidence scores for each detection
- 🎯 Real-time detection capabilities
- 👀 Handles occlusions and partial visibility
- ⚡ GPU acceleration support

### 3. **Comprehensive Reporting**
- 📝 **Text Reports**: Detailed statistics and metrics
- 💾 **CSV Export**: Historical data for analysis
- 🎨 **Color-coded Display**: Green (empty) / Red (occupied) / Yellow (Guiding Path)
- ⏱️ Real-time statistics display
- 📊 Occupancy rate tracking

### 4. **Advanced Features**
- ♿ Special parking space detection (yellow markings)
- 🔄 Video loop playback
- 📍 Precise coordinate tracking
- 🛡️ Robust error handling
- 📝 Comprehensive logging
- ⚙️ Centralized configuration

---

## 🚀 Quick Start

### Prerequisites

- Python 3.8 or higher
- pip package manager
- (Optional) CUDA-capable GPU for faster processing

### Installation

```bash
# 1. Clone the repository
git clone https://github.com/8harath/Car-Parking-Detection.git
cd Car-Parking-Detection

# Create virtual environment (recommended)
python -m venv venv

# Activate virtual environment
# On Windows:
venv\Scripts\activate
# On Unix or MacOS:
source venv/bin/activate

# 2. Install dependencies
pip install -r requirements.txt

# 3. Run the application
python run.py --camera [CAMERA_URL] 
```

### First Run

1. **Select Parking Spaces**: Click and drag to mark parking areas
2. **Save Layout**: Press `S` to save your parking space configuration
3. **Detect Vehicles**: Press `D` to run detection and generate reports
4. **View Results**: Check the `reports/` directory for outputs

**Keyboard Shortcuts:**
- `D` - Detect vehicles & generate reports
- `S` - Save parking layout
- `R` - Reset all selections
- `Z` - Undo last selection
- `Q` - Quit application

**What happens:**
1. The screen will show: "🚗 Detecting vehicles..."
2. The AI will analyze each parking space
3. Green spaces = Empty
4. Red spaces = Occupied
5. Reports are generated automatically

#### View Your Results (2 minutes)

The program creates several files for you:

**1. Visual Report** (`reports/parking_report_YYYYMMDD_HHMMSS.png`)
   - Open this image to see charts and statistics
   - Shows a colorful dashboard with graphs

**2. Text Report** (`reports/parking_report_YYYYMMDD_HHMMSS.txt`)
   - Open with any text editor
   - Contains detailed numbers and percentages

**3. CSV Data** (`data/parking_status.csv`)
   - Open with Excel or Google Sheets
   - Great for tracking over time


When you look at the visual report, you'll see
```
┌─────────────────────────────────────────┐
│  Parking Lot Image (with colors)        │
│  Green = Empty  |  Red = Occupied       │
└─────────────────────────────────────────┘

┌─────────────────────────────────────────┐
│  Bar Chart: Parking Statistics          │
│  Shows total, occupied, available       │
└─────────────────────────────────────────┘

┌─────────────────────────────────────────┐
│  Pie Charts: Space Distribution         │
│  Regular vs. Special parking            │
└─────────────────────────────────────────┘
```

The text report tells you:
- Total number of parking spaces
- How many are occupied
- How many are available
- Occupancy percentage
- Timestamp of detection

## 🔍 How It Works

### Architecture Overview

```
┌─────────────────┐
│   Input         │
│ (Image/Video)   │
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│ Space Selection │
│   (Manual UI)   │
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│ Image Processing│
│  (CV Pipeline)  │
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│ Vehicle Detection│
│    (YOLOv8)     │
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│Space Occupancy  │
│   Analysis      │
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│ Report Generation│
│ (Visual/CSV)    │
└─────────────────┘
```

### Detection Pipeline

1. **Pre-processing**
   - Grayscale conversion
   - Gaussian blur (noise reduction)
   - Adaptive thresholding
   - Median blur
   - Morphological operations (dilation)

2. **Vehicle Detection**
   - YOLOv8 inference on full image
   - Filter for vehicle classes (car, truck, bus, motorcycle)
   - Apply confidence threshold
   - Match detections to parking spaces

3. **Occupancy Analysis**
   - Extract each parking space region
   - Count non-zero pixels
   - Compare against threshold
   - Mark as occupied/empty

4. **Special Parking Detection**
   - Convert to HSV color space
   - Detect yellow markings
   - Identify horizontal/vertical lines
   - Mark as special parking space

---
