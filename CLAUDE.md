# XR Robotics Telemetry Visualization - Project Specification

## 1. Project Overview
This project is a technical assessment for an XR/Robotics engineering position.
The objective is to parse raw VR telemetry logs (camera parameters, 3D tracking data for the head, body, and hands) and visualize them using three distinct approaches:
1. **AR Video Overlay (`output_tracking.mp4`):** Projection of 3D skeletons onto a 2D first-person stereo video.
2. **Offline 3D Player (`skeleton_3d.html`):** An animated, downsampled HTML file for offline spatial analysis.
3. **Interactive Web Dashboard:** A Streamlit application for real-time, frame-by-frame 3D data streaming and inspection.

## 2. Directory Structure & Version Control
- `data/` - Contains raw inputs (`CameraRecord_*.mp4`, `trackingData_*.txt`). MUST be ignored in Git.
- `src/` - Python source modules:
  - `parser.py` - Core data extraction and normalization.
  - `visualize.py` - OpenCV video rendering and 3D-to-2D matrix projections.
  - `visualize_3d.py` - Plotly-based animated HTML generation.
  - `dashboard.py` - Streamlit web application.
- `main.py` - Single CLI entry point located in the root directory.
- `requirements.txt` - Dependencies (`numpy`, `opencv-python`, `plotly`, `streamlit`).
- `.gitignore` - Must ignore `data/*` (except `.gitkeep`), `venv/`, `__pycache__/`, and `*.mp4`.

## 3. Data Anomalies & Normalization (CRITICAL)
### 3.1. Parsing Tracking Data (`trackingData_*.txt`)
- **Camera Matrices:** Line 1 contains `cameraIntrinsics` `[fx, fy, cx, cy]` and `cameraExtrinsics` (extract the FIRST 4x4 matrix representing the left eye).
- **The Unity Comma Bug:** Spatial coordinate arrays (`p`) are corrupted due to Unity localization logic. Values appear as `"-0,3162,-0,4188,0,0014"`. 
  - *Fix:* Split the string by commas, group the resulting chunks into pairs, and join them with a dot `.` to reconstruct valid floats.
- **Time Synchronization:** Sync video frames and telemetry exclusively using `timeStampNs` converted to seconds relative to the start. Do not use linear array index mapping.

### 3.2. Coordinate Transformations
- Telemetry (`Head.pose`, `Body.joints`, `Hand`) is in absolute World Space.
- Pipeline for 2D projection: Calculate `Head_to_World` matrix -> Apply `Camera_to_Head` offset (Extrinsics) -> Compute `World_to_Camera` -> Project using `cameraIntrinsics`.

## 4. Feature Implementation Details
### 4.1. Video Overlay (`src/visualize.py`)
- Crop the Side-by-Side (SBS) stereo video to `width // 2` to isolate the left eye view.
- Draw the Head, Body, Left Hand, and Right Hand skeletons using `cv2.line` and `cv2.circle`.

### 4.2. Offline 3D Player (`src/visualize_3d.py`)
- Use `plotly.graph_objects` to create an animated 3D scatter plot (Scatter3d) with `frames` and `sliders`.
- **CRITICAL CONSTRAINT:** Downsample the parsed data (e.g., 200-250 frames). Rendering 21,000+ frames will result in an unopenable HTML file.
- Maintain `aspectmode='data'` to preserve accurate human proportions.

### 4.3. Interactive Dashboard (`src/dashboard.py`)
- Build a local web app using Streamlit.
- Load data into memory once using `@st.cache_data`.
- Provide a UI slider to select any frame index. Dynamically render the exact frame in a 3D Plotly chart.

## 5. CLI Interface (`main.py`)
Implement `argparse` to route execution based on subcommands:
- `python main.py video` -> executes `src/visualize.py` logic.
- `python main.py html` -> executes `src/visualize_3d.py` logic.
- `python main.py dashboard` -> uses `subprocess.run(["streamlit", "run", "src/dashboard.py"])`.

## 6. Architecture & Integrity
Maintain absolute mathematical purity. The right hand exhibits a spatial drift on the video overlay due to hardware calibration or IK algorithm offsets. Do not implement hardcoded pixel offsets (`x - 50`) to artificially align the visuals. Leave the pure math intact; the hardware drift will be addressed in the project documentation.
