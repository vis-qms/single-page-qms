# VIS-QMS Single-Page Enhanced

A comprehensive, single-page Queue Management System with all advanced features from the main FastAPI QMS application.

## ✨ Features

### 🎯 Core Functionality
- **Real-time People Counting**: Advanced YOLO-based person detection
- **Queue Wait Time Prediction**: ML-powered and rule-based estimation
- **Live Display**: Beautiful airport-style queue display
- **WebSocket Support**: Real-time updates with fallback polling

### 🤖 Advanced Detection Models
- **YOLOv8m/x**: Classic YOLO models (medium/extra-large)
- **YOLOv9e**: GELAN architecture for enhanced accuracy
- **YOLOv10x**: High precision detection
- **YOLOv11l/x**: Latest YOLO with excellent precision/recall balance
- **YOLOv12l/x**: State-of-the-art attention-centric models
- **RT-DETR-X**: Real-time DETR transformer for crowded scenes
- **DETR ResNet-50**: End-to-end transformer detection

### 🎛️ Enhanced Detection Pipeline
- **Test-Time Augmentation (TTA)**: Improved accuracy through augmentation
- **Secondary IoU Suppression**: Advanced duplicate removal
- **Geometric Filtering**: Min height/area ratio filtering
- **ROI Support**: Rectangle and polygon region of interest
- **Confidence Thresholding**: Configurable detection confidence

### 🎯 Tracking & Stabilization
- **IoU-based Tracking**: Maintain person identities across frames
- **Count Stabilization**: EMA and Rolling Average smoothing
- **Max Delta Limiting**: Prevent sudden count jumps
- **People Adjustment**: Fine-tune final count with offset

### 📐 Advanced Cropping
- **Rectangle Cropping**: Percentage-based region selection
- **Polygon Cropping**: Visual polygon editor with drag-and-drop
- **Live Preview**: Real-time overlay visualization
- **Normalized Coordinates**: Resolution-independent settings

### 🔮 Wait Time Prediction
- **ML Prediction**: Airport check-in queue predictor model
- **Rule-based Fallback**: Configurable per-person time ranges
- **Hour-aware**: Time-of-day consideration for ML model
- **Feature Engineering**: Advanced features for accurate prediction

### 🎨 Enhanced Display
- **Airport-style Theming**: Professional dark theme
- **Color-coded Wait Times**: Green/Yellow/Red status indicators
- **Responsive Design**: Scales from mobile to large displays
- **Status Indicators**: Connection and detection status
- **Smooth Animations**: Polished user experience

### 📊 Telemetry & History
- **Detection History**: Store last 200 detection records
- **Performance Metrics**: Inference time tracking
- **Model Analytics**: Per-model performance statistics
- **Export Capabilities**: JSON configuration export/import

### 🔧 Configuration Management
- **Password Encoding**: Base64 encoded password storage
- **File-based Config**: Save/load configurations to disk
- **Hot Reloading**: Apply settings without restart
- **Validation**: Input validation and error handling

## 🚀 Quick Start

### Prerequisites
- Python 3.8+
- CUDA-capable GPU (optional, for better performance)
- Webcam or IP camera

### Installation

1. **Clone and navigate:**
```bash
cd single-page-qms
```

2. **Install dependencies:**
```bash
pip install -r requirements.txt
```

3. **Run the application:**
```bash
python -m uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
```

4. **Access the interfaces:**
- Configuration: http://localhost:8000
- Queue Display: http://localhost:8000/display
- API Docs: http://localhost:8000/docs

## 📖 Usage Guide

### Basic Setup

1. **Configure Camera:**
   - Select connection type (Webcam/NVR/Custom URL)
   - Enter connection details (IP, credentials, etc.)
   - Test connection with live preview

2. **Detection Settings:**
   - Choose detection model (YOLOv11x recommended)
   - Set confidence threshold (0.5 default)
   - Configure image size and processing options

3. **Region of Interest:**
   - Enable rectangle or polygon cropping
   - Use visual editor to define detection area
   - Preview changes in real-time

4. **Start Detection:**
   - Click "Start Detection" button
   - Monitor live metrics and status
   - Open display on secondary screen

### Advanced Configuration

#### Tracking & Stabilization
```json
{
  "tracker": {
    "enabled": true,
    "ttl": 2,
    "iou_thr": 0.40
  },
  "count_stabilization": {
    "method": "EMA",
    "ema_alpha": 0.65,
    "max_delta_per_detection": 1
  }
}
```

#### Polygon ROI
```json
{
  "polygon_cropping": {
    "enabled": true,
    "points": [
      {"x": 0.2, "y": 0.1},
      {"x": 0.8, "y": 0.1},
      {"x": 0.9, "y": 0.9},
      {"x": 0.1, "y": 0.9}
    ]
  }
}
```

#### Advanced Detection
```json
{
  "runtime": {
    "use_tta": true,
    "max_det": 100,
    "secondary_iou": 0.70,
    "min_height_ratio": 0.018,
    "min_area_ratio": 0.00015
  }
}
```

## 🎛️ API Endpoints

### Core Endpoints
- `GET /api/status` - Live system status
- `POST /api/start` - Start detection
- `POST /api/stop` - Stop detection
- `GET /api/config` - Get configuration
- `POST /api/config` - Update configuration

### Enhanced Endpoints
- `GET /api/display_data` - Display-optimized data
- `GET /api/history` - Detection history
- `DELETE /api/history` - Clear history
- `GET /api/models` - Available models
- `WS /ws/status` - WebSocket real-time updates

### Configuration Management
- `POST /api/config/save` - Save config to file
- `GET /api/config/load` - Load config from file
- `GET /api/diagnose` - System diagnostics

## 🎨 Display Customization

### URL Parameters
- `?ws=true` - Enable WebSocket updates
- `?refresh=1000` - Polling interval (ms)
- `?size=large` - Display size preset

### Theme Options
- **Dark (Airport Standard)**: Professional airport styling
- **Light (Bright)**: High visibility bright theme
- **High Contrast**: Accessibility-focused theme

## 🔧 Troubleshooting

### Common Issues

1. **Camera Connection Failed:**
   - Check IP address and credentials
   - Verify network connectivity
   - Test with VLC or similar tool first

2. **Model Loading Errors:**
   - Ensure sufficient disk space
   - Check internet connection for downloads
   - Verify CUDA installation for GPU models

3. **Performance Issues:**
   - Reduce image size (imgsz parameter)
   - Disable TTA and advanced features
   - Use smaller models (YOLOv8m vs YOLOv12x)

4. **Polygon Editor Not Working:**
   - Ensure JavaScript is enabled
   - Clear browser cache
   - Check browser console for errors

### Debug Mode
```bash
# Enable debug logging
export LOG_LEVEL=DEBUG
python -m uvicorn app.main:app --reload --log-level debug
```

## 📊 Performance Optimization

### GPU Acceleration
- Install CUDA toolkit
- Enable half-precision (FP16)
- Use appropriate batch sizes

### CPU Optimization
- Use smaller models (YOLOv8m)
- Reduce image resolution
- Increase detection intervals

### Memory Management
- Limit detection history size
- Clear browser cache regularly
- Monitor system resources

## 🔒 Security Notes

- Passwords are Base64 encoded (not encrypted)
- Use HTTPS in production environments
- Restrict network access appropriately
- Regular security updates recommended

## 🤝 Contributing

This enhanced single-page QMS includes all features from the main FastAPI QMS application:

- ✅ Advanced detection models (YOLOv8-12, RT-DETR, DETR)
- ✅ Enhanced detection pipeline with TTA and filtering
- ✅ IoU-based tracking system
- ✅ EMA/Rolling average stabilization
- ✅ Polygon cropping with visual editor
- ✅ ML-based wait time prediction
- ✅ Professional display styling
- ✅ WebSocket real-time updates
- ✅ Configuration management
- ✅ Detection history and telemetry
- ✅ Comprehensive API endpoints

## 📄 License

This project maintains the same license as the original FastAPI QMS application.

---

**Built with ❤️ for efficient queue management**