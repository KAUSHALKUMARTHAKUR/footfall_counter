# Footfall Counter using Computer Vision

A real-time people counting system using YOLOv8 object detection and centroid tracking, designed with an "escalator mode" approach for accurate entry/exit counting through separate detection lines.

**Author:** Kaushal Kumar Thakur  
**Date:** November 2025

---

## 📹 Demo Videos

- **Input Video:** [View on Google Drive](https://drive.google.com/file/d/1Ng1SEHwMbyMV6vacLJQpBDLXMAoe3tWH/view?usp=sharing)
- **Output Video:** [View on Google Drive](https://drive.google.com/file/d/13L-UAlLP1k_8Da62AMRSIUACy14EIe1R/view?usp=sharing)

---

## 🎯 Project Overview

This project implements an intelligent footfall counter that tracks people entering and exiting a designated area using computer vision techniques. The system uses state-of-the-art YOLOv8 object detection combined with centroid tracking to accurately count individuals crossing predefined entry and exit lines.

### Key Features

- ✅ **Dual Line Detection:** Separate entry and exit lines for accurate bidirectional counting
- ✅ **Real-time Processing:** Live video feed processing with FPS monitoring
- ✅ **Interactive ROI Selection:** User-friendly interface to define counting zones
- ✅ **Persistent Configuration:** Save and load ROI configurations for reuse
- ✅ **Visual Feedback:** Color-coded tracking states (Orange=Tracking, Green=Entered, Red=Exited)
- ✅ **Comprehensive Statistics:** Real-time display of entries, exits, total count, and people currently inside
- ✅ **Webcam Support:** Works with both video files and live webcam feeds

---

## 🧠 Approach & Methodology

### 1. **Object Detection (YOLOv8)**
The system uses YOLOv8n (nano version) for fast and accurate person detection. YOLOv8 is chosen for its:
- High accuracy in person detection
- Real-time processing capabilities
- Low computational requirements (especially the nano model)
- Pre-trained on COCO dataset with robust person detection

### 2. **Centroid Tracking**
A custom centroid tracker maintains identity of detected persons across frames:
- **Centroid Calculation:** Computes center points of bounding boxes
- **Object Registration:** Assigns unique IDs to new detections
- **Distance-based Matching:** Uses Euclidean distance to match objects across frames
- **Disappearance Handling:** Maintains objects for up to 50 frames after disappearance
- **Maximum Distance Threshold:** 80 pixels to prevent false matches

### 3. **Escalator Mode Counting Logic**
The core counting mechanism is inspired by escalator counting systems:

#### Line Crossing Detection
```
Entry Line (Green): ─────────────────
                    ▲
                   tolerance zone (±15px)
                    
Exit Line (Red):   ─────────────────
                    ▲
                   tolerance zone (±15px)
```

**Counting Algorithm:**
1. **Zone Definition:** Each line has a tolerance zone of ±15 pixels vertically
2. **Centroid Check:** For each tracked object, check if centroid falls within:
   - Line's horizontal range (x1 < cx < x2)
   - Line's vertical tolerance zone (y ± tolerance)
3. **One-time Counting:** Each object ID is counted only once per line
4. **Separate Lists:** Maintains distinct lists for entry and exit crossings

**Metrics Calculated:**
- **Entries:** Count of unique IDs that crossed the entry line
- **Exits:** Count of unique IDs that crossed the exit line
- **Total:** Sum of entries and exits (total footfall)
- **Inside:** Difference between entries and exits (current occupancy)

### 4. **Visual Processing Pipeline**
```
Video Frame → YOLOv8 Detection → Centroid Tracking → 
Line Crossing Check → Visual Rendering → Output Display
```

---

## 🔧 Technical Implementation

### Detection & Tracking Parameters

| Parameter | Value | Purpose |
|-----------|-------|---------|
| Confidence Threshold | 0.4 | Minimum detection confidence |
| Max Disappeared Frames | 50 | Frames before deregistering object |
| Max Distance | 80px | Maximum centroid movement between frames |
| Tolerance Zone | ±15px | Vertical tolerance for line crossing |
| Class Filter | Person (0) | COCO class for person detection |

### Color Coding

- 🟢 **Green:** Entry line and tracked persons who have entered
- 🔴 **Red:** Exit line and tracked persons who have exited
- 🟠 **Orange:** Persons currently being tracked (not yet crossed a line)

---

## 📋 Dependencies

### Python Version
- Python 3.8 or higher

### Required Libraries
```
opencv-python>=4.8.0
numpy>=1.24.0
ultralytics>=8.0.0
argparse (built-in)
pathlib (built-in)
json (built-in)
```

### System Requirements
- **Minimum:** 4GB RAM, Intel i5 or equivalent
- **Recommended:** 8GB RAM, GPU support (CUDA) for faster processing
- **Storage:** ~50MB for YOLOv8n model weights

---

## 🚀 Setup Instructions

### 1. Clone or Download the Project
```bash
cd your_project_directory
```

### 2. Create Virtual Environment (Recommended)
```bash
# Windows
python -m venv venv
venv\Scripts\activate

# Linux/Mac
python3 -m venv venv
source venv/bin/activate
```

### 3. Install Dependencies
```bash
pip install opencv-python numpy ultralytics
```

The YOLOv8n model will be automatically downloaded on first run (~6MB).

### 4. Prepare Your Video
Place your input video file in the project directory or prepare to use a webcam.

---

## 💻 Usage Guide

### Step 1: Select ROI (First Time Setup)

Run the ROI selection tool to define your entry and exit lines:

```bash
python footfall_counter.py --video input.mp4 --select-roi
```

**Interactive Instructions:**
1. **Entry Line (Green):** Click 2 points horizontally across the entry area → Press `n`
2. **Exit Line (Red):** Click 2 points horizontally across the exit area → Press `s` to save

**Controls:**
- `n` - Next (after completing entry line)
- `s` - Save ROI configuration
- `r` - Reset current line
- `q` - Quit without saving

The configuration is saved to `roi_config.json` for reuse.

### Step 2: Process Video

Using saved ROI:
```bash
python footfall_counter.py --video input.mp4 --load-roi roi_config.json --output output/result.mp4
```

### Common Usage Examples

**Process video with saved ROI:**
```bash
python footfall_counter.py --video my_video.mp4 --load-roi roi_config.json
```

**Use webcam with custom confidence:**
```bash
python footfall_counter.py --webcam --select-roi --confidence 0.5
```

**Process without display (headless mode):**
```bash
python footfall_counter.py --video input.mp4 --load-roi roi_config.json --no-display
```

**Use custom YOLO model:**
```bash
python footfall_counter.py --video input.mp4 --load-roi roi_config.json --model yolov8s.pt
```

### Command Line Arguments

| Argument | Type | Description | Default |
|----------|------|-------------|---------|
| `--video` | str | Path to input video file | None |
| `--webcam` | flag | Use webcam instead of video file | False |
| `--output` | str | Path to save output video | `output/counted.mp4` |
| `--model` | str | YOLOv8 model variant | `yolov8n.pt` |
| `--select-roi` | flag | Launch ROI selection interface | False |
| `--load-roi` | str | Load ROI from JSON file | None |
| `--save-roi` | str | Save ROI to JSON file | `roi_config.json` |
| `--confidence` | float | Detection confidence threshold | 0.4 |
| `--no-display` | flag | Disable real-time display | False |

---

## 📊 Output Information

### Real-time Display

The system shows:
1. **Video Feed:** Original video with overlays
2. **Entry Line:** Green horizontal line
3. **Exit Line:** Red horizontal line
4. **Bounding Boxes:** Color-coded boxes around detected persons
5. **Object IDs:** Unique identifier for each tracked person
6. **Status Labels:** TRACKING, ENTERED, or EXITED
7. **Statistics Panel:**
   - ENTRIES: Total people entered
   - EXITS: Total people exited
   - TOTAL: Total footfall (entries + exits)
   - INSIDE: Current occupancy (entries - exits)
8. **FPS Counter:** Real-time processing speed

### Terminal Output

```
[INFO] Loading YOLOv8: yolov8n.pt
[INFO] Escalator Mode - Tolerance: ±15px
[INFO] Processing...

✓ [ENTRY] ID 5 crossed entry line
✓ [EXIT] ID 3 crossed exit line
[95.2%] Entries:15 Exits:8

======================================================================
FINAL RESULTS - ESCALATOR MODE
======================================================================
Frames: 1250 | Time: 45.32s | FPS: 27.58
✓ ENTRIES: 15
✓ EXITS: 8
✓ TOTAL: 23
✓ INSIDE: 7
======================================================================
```

---

## 🎥 Video Source

The demo uses a surveillance-style video showing pedestrian traffic in a corridor or entrance area. This type of footage is ideal for footfall counting as it provides:
- Clear overhead or angled perspective
- Consistent lighting conditions
- Defined entry and exit points
- Multiple people walking in both directions

**Input Video:** Available at [Google Drive Link](https://drive.google.com/file/d/1Ng1SEHwMbyMV6vacLJQpBDLXMAoe3tWH/view?usp=sharing)

**Output Video:** Processed result with visualizations at [Google Drive Link](https://drive.google.com/file/d/13L-UAlLP1k_8Da62AMRSIUACy14EIe1R/view?usp=sharing)

---

## 🔍 Counting Logic Explained

### Why "Escalator Mode"?

The escalator mode is inspired by reliable people counting systems used in:
- Shopping malls (escalator traffic)
- Public transportation (turnstiles)
- Retail stores (entrance counters)

### Advantages of This Approach

1. **Bidirectional Counting:** Separate lines prevent confusion between entry/exit
2. **Tolerance Zones:** ±15px buffer accounts for detection jitter
3. **One-time Counting:** Each person counted only once per line (prevents duplicates)
4. **No Direction Analysis:** Simpler than trajectory-based methods, more robust
5. **ID Persistence:** Centroid tracking maintains identity across frames

### Handling Edge Cases

- **Occlusion:** Tracker maintains IDs for 50 frames during brief occlusions
- **False Detections:** Confidence threshold (0.4) filters weak detections
- **Fast Movement:** Max distance threshold (80px) prevents ID swaps
- **Crowding:** Centroid-based matching handles overlapping bounding boxes

### Accuracy Considerations

**Factors Improving Accuracy:**
- High-quality video with clear view of counting area
- Proper placement of entry/exit lines in uncrowded zones
- Adequate lighting conditions
- Appropriate camera angle (overhead or 45-degree angle)

**Potential Limitations:**
- Heavy occlusion may cause missed counts
- Very fast movement might exceed max distance threshold
- Extremely crowded scenes may have detection overlaps
- Line placement in high-traffic zones may cause ambiguity

---

## 🛠️ Troubleshooting

### Issue: "Cannot open video"
- **Solution:** Check video file path and format (use MP4, AVI, MOV)

### Issue: Low FPS / Slow processing
- **Solution:** Use lighter model (`yolov8n.pt`) or reduce video resolution
- **Solution:** Enable GPU acceleration if available

### Issue: Missed detections
- **Solution:** Lower confidence threshold: `--confidence 0.3`
- **Solution:** Improve lighting in video

### Issue: False counts
- **Solution:** Increase confidence threshold: `--confidence 0.5`
- **Solution:** Adjust tolerance zone in code (line 180)

### Issue: ROI not loading
- **Solution:** Ensure `roi_config.json` exists in the same directory
- **Solution:** Re-run ROI selection with `--select-roi`

---

## 📁 Project Structure

```
footfall_counter/
│
├── footfall_counter.py       # Main script
├── roi_config.json           # ROI configuration (generated)
├── README.md                 # This file
│
├── output/                   # Output videos directory
│   └── counted.mp4
│
└── yolov8n.pt               # YOLO model (auto-downloaded)
```

---

## 🎓 Educational Value

This project demonstrates:
- **Computer Vision:** Object detection and tracking
- **Deep Learning:** YOLOv8 neural network inference
- **Algorithm Design:** Centroid tracking and line crossing detection
- **Software Engineering:** Modular design, configuration management
- **User Interface:** Interactive ROI selection with OpenCV

---

## 🚀 Future Enhancements

Potential improvements:
- [ ] Multi-line support for complex layouts
- [ ] Heatmap visualization of high-traffic areas
- [ ] CSV export of timestamp-based counts
- [ ] Email/SMS alerts for occupancy thresholds
- [ ] Deep SORT tracking for improved ID persistence
- [ ] Cloud deployment for remote monitoring
- [ ] Mobile app integration
- [ ] Analytics dashboard with historical data

---

## 📝 License

This project is developed as part of an AI assignment for educational purposes.

---

## 👤 Author

**Kaushal Kumar Thakur**  
November 2025

---

## 🙏 Acknowledgments

- **Ultralytics:** YOLOv8 framework
- **OpenCV:** Computer vision library
- **COCO Dataset:** Pre-trained model weights

---

## 📞 Support

For issues, questions, or contributions, please review the code documentation and this README. The implementation follows best practices for computer vision-based people counting systems.

---

**Note:** This system is designed for educational and demonstration purposes. For production deployment, consider additional validation, error handling, and performance optimization based on specific use case requirements.

