# Cheating Monitor System
An online examination proctoring and cheating behavior recognition system based on computer vision. It supports three detection modes: **image, video and real-time camera stream**. The system identifies suspicious behaviors including abnormal head posture, gaze deviation, mobile phone usage, leaving seat and multiple people gathering. It generates evidence screenshots and Excel reports via threshold-based alarm mechanism.

---

## 1. Features

| Category | Description |
|---|---|
| Detection Items | Head posture (6 directions), pupil/gaze direction (4 directions), mobile phone detection, seat absence, multiple people, eye closure |
| Input Sources | Images, local video files, real-time camera |
| Controls | Start / Pause / Resume / Stop / Clear records |
| Threshold Settings | Independent time thresholds (seconds) for 5 violation types, fully adjustable |
| Performance | Video frame skipping (3× acceleration by default), full-frame detection optional |
| Calibration | 30-frame explicit camera calibration, silent auto-calibration for first video frame, two-step auto-calibration for images |
| Output | Automatic violation screenshot archiving + detailed & statistical Excel reports |
| Interface | Dual before/after detection display panels, 5+1 statistical cards, color-coded logs, event detail tables |
| Stability | Inference runs in independent QThread, no UI lag, automatic state reset between sessions |

---

## 2. Project Structure

```
Cheating-Surveillance-System/
├── cheating_monitor_system.py    # Main GUI program
├── eye_movement.py               # Pupil & gaze direction detection module
├── head_pose.py                  # Head pose estimation module
├── mobile_detection.py           # Custom mobile phone detection module
├── model/
│   └── shape_predictor_68_face_landmarks.dat    # dlib 68 facial landmark model
├── results/                      # Runtime detection output
│   └── session_YYYYMMDD_HHMMSS/
│       ├── head_*.png            # Head violation screenshots
│       ├── eye_*.png             # Eye gaze violation screenshots
│       ├── mobile_*.png           # Mobile phone screenshots
│       ├── absence_*.png         # Seat absence screenshots
│       ├── crowd_*.png          # Multiple people screenshots
│       └── report_*.xlsx         # Exported Excel detection report
└── README.md
```

---

## 3. Installation

### 3.1 Python Environment
Python 3.7 or higher is required. Virtual environment recommended:
```bash
python -m venv venv
# Windows
venv\Scripts\activate
# Linux / macOS
source venv/bin/activate
```

### 3.2 Install Dependencies
```bash
pip install PyQt5 opencv-python numpy openpyxl dlib
```

**dlib Installation Note**:
If `pip install dlib` fails to compile, use precompiled wheel package:
```bash
pip install dlib-bin
```

### 3.3 Download Model File
Place the dlib 68 facial landmark model in project root or `model/` folder:
- Download link: <http://dlib.net/files/shape_predictor_68_face_landmarks.dat.bz2>
- Extract to get `shape_predictor_68_face_landmarks.dat` (~100 MB)

The program automatically searches paths below:
1. Current working directory
2. `./model/` subfolder
3. Directory of Python script files
4. `model/` folder under script directory

---

## 4. Quick Start
```bash
python cheating_monitor_system.py
```

### Typical Workflow
1. **Select Detection Source** — Click `① Image Detection` / `② Video Detection` / `③ Camera Detection` on left panel. Original frame will display immediately.
2. **Adjust Thresholds (Optional)** — Default values adapt to most scenarios.
3. Click **▶ Start Detection**
   - Camera mode: 30-frame head calibration (~1s), keep face forward
   - Image & video mode: automatic silent calibration on first frame
4. Use **⏸ Pause** or **■ Stop** at any time
5. Click **💾 Save Results** to export Excel report

---

## 5. Interface Layout
```
┌───────────────┬─────────────────────────────────┬──────────────┐
│ Control Panel │          Detection Display      │ Statistics   │
│               │  ┌────────────┬────────────┐   │ ┌────┬────┐  │
│ Detection Mode│  │ Before Detect│ After Detect│ │ │Head│Eye │  │
│ ① Image       │  │ (Original)   │ (Annotated) │ │ ├────┼────┤  │
│ ② Video       │  └────────────┴────────────┘   │ │Phone│Away │ │
│ ③ Camera      │                                 │ ├────┼────┤  │
│               │ Real-time Status                │ │Crowd│Total│ │
│ Controls      │ [Head✓] [Eye✓] [Phone✓]        │ └────┴────┘  │
│ ▶ Start       │                                 │              │
│ ⏸ Pause       │                                 │ Event Log    │
│ ■ Stop        │                                 │ Time/Type/Desc│
│               │                                 │              │
│ Alert Threshold│                                │ Color Logs   │
│ Head: 3.0s    │                                 │              │
│ Eye: 3.0s     │                                 │              │
│ Phone: 3.0s   │                                 │              │
│ Away: 2.0s    │                                 │              │
│ Crowd: 1.0s   │                                 │              │
│ Frame Skip: 2 │                                 │              │
│               │                                 │              │
│ Export Output │                                 │              │
│ 💾 Save Excel  │                                 │              │
│ 🗑 Clear Logs │                                 │              │
└───────────────┴─────────────────────────────────┴──────────────┘
```

### Status Indicator Colors
| Color | Meaning |
|---|---|
| 🟢 Green | Normal status (watching screen, no phone detected) |
| 🔴 Red | Violation detected |
| 🟠 Orange | Eyes closed (not a violation, disables gaze detection) |

---

## 6. Parameter Configuration

### Alert Thresholds
| Parameter | Default Value | Description |
|---|---|---|
| Head Threshold | 3.0 s | Trigger violation when head deviates continuously beyond duration |
| Eye Threshold | 3.0 s | Trigger violation when gaze deviates continuously beyond duration |
| Phone Threshold | 3.0 s | Trigger violation when mobile phone detected continuously |
| Absence Threshold | 2.0 s | Trigger violation when no face detected in frame |
| Crowd Threshold | 1.0 s | Trigger violation when ≥2 faces appear continuously |

Smaller threshold = higher sensitivity; larger threshold = looser judgment.
Recommended settings:
- Strict exam scenarios: 1.0–2.0s for all items
- Normal scenarios: use default values
- Relaxed scenarios: set head & eye thresholds to 5.0s

### Video Settings
| Parameter | Default | Description |
|---|---|---|
| Video Frame Skip | 2 frames | Skip N frames per detection. 0=full frame, 2=3× speed, 5=6× speed |

Frame skip does not affect image and camera detection.

---

## 7. Violation Definition

### 5 Violation Types
| Key | Name | Trigger Condition | Risk Level |
|---|---|---|---|
| `head` | Head Abnormality | Head turns Left/Right/Up/Down/Tilted | Medium |
| `eye` | Gaze Deviation | Pupil gaze Left/Right/Up/Down | Medium |
| `mobile` | Mobile Phone Use | Mobile phone detected in frame | High |
| `absence` | Seat Absence | No human face in frame | Critical |
| `crowd` | Multiple People | ≥2 faces detected simultaneously | Extremely Critical |

### Non-Violation Status
| Status | Handling Rule |
|---|---|
| Looking at Screen | Normal, no counting |
| Eyes Closed | Orange prompt only, excluded from eye violations to avoid blink false alarms |

### Threshold Logic
- Head/gaze/phone: long 3s tolerance for natural brief movements
- Absence & crowd: short 1–2s threshold for severe abnormal behaviors

---

## 8. Output Description

### Session Folder Structure
A separate timestamped folder is created for each detection session:
```
results/session_20260421_143012/
├── head_Looking Left_20260421_143045_123456.png
├── eye_Looking Down_20260421_143112_789012.png
├── mobile_20260421_143201_345678.png
├── absence_20260421_143312_111222.png
├── crowd_20260421_143412_333444.png
└── report_20260421_143500.xlsx
```

Filename format: `{violation_type}_{timestamp}.png`

### Excel Report
Contains two worksheets:
**Sheet 1: Event Details**
| No. | Time | Type | Description | Screenshot Path |
|---|---|---|---|---|
| 1 | 2026-04-21 14:30:45 | head | Head deviated left | results/.../head_*.png |

**Sheet 2: Statistical Summary**
| Violation Type | Occurrence Count |
|---|---|
| Head Abnormality | 3 |
| Gaze Deviation | 5 |
| Mobile Phone | 1 |
| Seat Absence | 0 |
| Multiple People | 0 |
| Total | 9 |

### Log Levels
| Level | Color | Usage |
|---|---|---|
| `INFO` | Blue | Normal operation records |
| `WARN` | Yellow | Recoverable minor exceptions |
| `ERROR` | Red | Critical runtime errors |
| `ALARM` | Bright Red | Cheating violation alerts |

---

## 9. Technical Architecture

### Module Functions
- `eye_movement.py`: Otsu threshold + polygon mask gaze detection, EAR eye closure detection, state smoothing
- `head_pose.py`: dlib 68 landmarks + solvePnP pose estimation, dynamic camera matrix, hysteresis filtering
- `mobile_detection.py`: Custom mobile object detection (YOLO-based)
- `cheating_monitor_system.py`: PyQt5 GUI & triple-thread framework

### Multi-Thread Workflow
```
 UI Thread           PreviewThread         DetectionThread
   │                    │                      │
   ├─ Select Source ───>│                      │
   │                    │── Frame Read ───> Display
   │                    │                      │
   ├─ Start Detect ──stop>│                      │
   │                    X                  ───>│ Detect & Alarm ──> UI
   │                                           │
   ├─ Pause/Resume ───────────────────────────>│
   │                                           │
   └─ Stop ───────────────────────────────────>X
```

### Calibration Mechanism
- Camera: 30-frame explicit calibration, keep face forward when prompted
- Video: Use first-frame pose as baseline reference
- Image: Two-step self-calibration to judge relative head direction

### Session Isolation
All buffers and historical states reset automatically when restarting detection, preventing cross-session data interference.

### Signal Transmission
```
DetectionThread
   │
   ├─ frame_ready(before, after)      → Update dual display
   ├─ status_update(head, eye, mobile) → Update status indicators
   ├─ cheat_detected(type, detail, img)→ Count, save screenshot, write report & log
   ├─ log_message(level, text)         → Update color interface logs
   └─ finished_signal                  → Restore UI button status
```

---

## 10. FAQ

### Q1: `FileNotFoundError: shape_predictor_68_face_landmarks.dat not found`
Place model file into valid search paths shown in installation guide.

### Q2: dlib installation failed
Install precompiled package:
```bash
pip install dlib-bin
```

### Q3: Camera cannot open
- Close other apps occupying camera (Zoom, meeting software)
- Use `cv2.CAP_DSHOW` backend on Windows
- Enable camera permission in macOS system settings

### Q4: Video plays too slowly
Increase frame skip value to speed up processing.

### Q5: Frequent false head violation alerts
Keep face aligned during calibration. Raise head threshold to 5.0s.

### Q6: Frequent multiple-person false alarms
Remove background people, increase crowd threshold to 3.0s.

### Q7: Can thresholds be changed during detection?
New values take effect in next detection start. Real-time dynamic adjustment requires code modification.

### Q8: Can it run without openpyxl?
Basic detection works, but Excel export function will be unavailable.

### Q9: Will closed eyes trigger continuous alarms?
No. Eye closure only shows orange prompt, no counting or alarm logs.

---

## 11. Expansion Functions
- Record annotated detection videos with `cv2.VideoWriter`
- Support multiple camera selection
- Real-time dynamic threshold update via signal slots
- Push violation alerts to management platform via WebSocket/HTTP
- Add face recognition to prevent exam impersonation
- Multi-channel concurrent remote online proctoring

---

## 12. Acknowledgements
- dlib — <http://dlib.net/>
- OpenCV — <https://opencv.org/>
- PyQt5 — <https://www.riverbankcomputing.com/software/pyqt/>

This system is fully reconstructed based on [Cheating-Surveillance-System](https://github.com/PrabhakarVenkat/Cheating-Surveillance-System) detection pipeline.

