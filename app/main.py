
from fastapi import FastAPI, Query, WebSocket, WebSocketDisconnect, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import StreamingResponse, FileResponse, JSONResponse, HTMLResponse
from pydantic import BaseModel
import os, cv2, time, json, base64
from typing import Dict, Any, Optional, List
from datetime import datetime
import numpy as np
from .state import STATE

app = FastAPI(title="VIS-QMS (No Streamlit)")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

static_dir = os.path.join(os.path.dirname(__file__), "..", "frontend")
app.mount("/static", StaticFiles(directory=static_dir), name="static")

@app.on_event("startup")
def _start():
    print("🚀 Server startup - detection will start only when user clicks 'Start Detection'")
    # Don't auto-start detection - wait for user to click Start Detection

@app.on_event("shutdown")
def _stop():
    STATE.stop()

@app.get("/")
def root():
    index = os.path.join(static_dir, "index.html")
    if os.path.exists(index):
        return FileResponse(index)
    return HTMLResponse("<h2>VIS-QMS</h2><p>Backend running.</p>")

@app.get("/display")
def display_page():
    html = os.path.join(static_dir, "display.html")
    if os.path.exists(html):
        return FileResponse(html)
    return HTMLResponse("<h2>Display page not found</h2>", status_code=404)

@app.get("/api/config")
def get_config():
    return JSONResponse(STATE.get_config())

@app.post("/api/config")
async def set_config(request: Request):
    payload = await request.json()
    old_config = STATE.get_config()
    ok = STATE.save_config(payload)
    
    # Check if connection changed - if so, restart detection if running
    if ok:
        old_conn = old_config.get("connection", {})
        new_conn = payload.get("connection", {})
        if old_conn != new_conn and STATE.running:
            print("🔄 Connection config changed while running - restarting detection...")
            # Don't actually stop/start, just let the worker loop detect the change
    
    return {"ok": ok}

@app.get("/api/display_data")
def get_display_data():
    """Enhanced display data endpoint matching main app"""
    cfg = STATE.get_config()
    with STATE.lock:
        wait_time_minutes = STATE.wait_time / 60.0
        return {
            "gate_id": cfg.get("gate_id", "GATE-01"),
            "gate_name": cfg.get("gate_name", "GATE"),
            "gate_number": cfg.get("gate_number", "01"),
            "people_count": STATE.people_count,
            "wait_time": wait_time_minutes,
            "wait_time_seconds": int(STATE.wait_time),
            "last_updated": STATE.last_frame_ts,
            "model": cfg.get("runtime", {}).get("selected_model", "YOLOv11x"),
            "display_config": cfg.get("display_config", {}),
            "connected": STATE.connected,
            "running": STATE.running,
            "debug": {
                "enabled": STATE.debug_enabled,
                "probability": STATE.debug_probability,
                "images_saved": STATE.debug_images_saved
            },
            "fps": {
                "actual_fps": STATE.actual_fps,
                "configured_fps": STATE.configured_fps,
                "fps_warning": STATE.fps_warning_active,
                "fps_percentage": round((STATE.actual_fps / STATE.configured_fps * 100), 1) if STATE.configured_fps > 0 else 0
            }
        }

@app.post("/api/start")
def api_start():
    print("🚀 API: Start detection requested")
    STATE.start()
    cfg = STATE.get_config()
    conn_type = cfg.get("connection", {}).get("connection_type", "Unknown")
    print(f"🚀 API: Detection started - running={STATE.running}, connected={STATE.connected}, type={conn_type}")
    return {"running": STATE.running, "connected": STATE.connected}

@app.post("/api/stop")
def api_stop():
    print("🛑 API: Stop detection requested")
    STATE.stop()
    print(f"🛑 API: Detection stopped - running={STATE.running}, connected={STATE.connected}")
    return {"running": STATE.running, "connected": STATE.connected}

@app.get("/api/status")
def api_status():
    cfg = STATE.get_config()
    with STATE.lock:
        return {
            "gate_id": cfg.get("gate_id", "GATE-01"),
            "gate_name": cfg.get("gate_name", "GATE"),
            "gate_number": cfg.get("gate_number", "01"),
            "connected": STATE.connected,
            "running": STATE.running,
            "people_count": STATE.people_count,
            "raw_people_count": getattr(STATE, "_raw_people_count", 0),
            "wait_time": STATE.wait_time,
            "wait_minutes": STATE.wait_time / 60.0,
            "wait_seconds": int(STATE.wait_time),
            "last_frame_ts": STATE.last_frame_ts,
            "last_error": getattr(STATE, "last_error", None),
            "total_detections": STATE.total_detections,
            "average_inference_time": STATE.average_inference_time,
            "total_cycle_time": getattr(STATE, "total_cycle_time", 0.0),
            "model": cfg.get("runtime", {}).get("selected_model", "YOLOv11x"),
            "confidence": cfg.get("runtime", {}).get("confidence_threshold", 0.5),
            "display_config": cfg.get("display_config", {}),
            "debug": {
                "enabled": STATE.debug_enabled,
                "probability": STATE.debug_probability,
                "images_saved": STATE.debug_images_saved
            },
            "fps": {
                "actual_fps": STATE.actual_fps,
                "configured_fps": STATE.configured_fps,
                "fps_warning": STATE.fps_warning_active,
                "fps_percentage": round((STATE.actual_fps / STATE.configured_fps * 100), 1) if STATE.configured_fps > 0 else 0
            }
        }

@app.get("/api/fps_status")
def get_fps_status():
    """Get current FPS monitoring status and camera performance"""
    with STATE.lock:
        fps_percentage = round((STATE.actual_fps / STATE.configured_fps * 100), 1) if STATE.configured_fps > 0 else 0
        return {
            "actual_fps": STATE.actual_fps,
            "configured_fps": STATE.configured_fps,
            "fps_warning": STATE.fps_warning_active,
            "fps_percentage": fps_percentage,
            "message": f"Camera delivering {STATE.actual_fps:.1f} FPS ({fps_percentage}% of configured {STATE.configured_fps} FPS)" if STATE.configured_fps > 0 else "FPS monitoring inactive"
        }


@app.get("/preview.mjpg")
def mjpeg_preview(
    q: int = Query(75, ge=10, le=95),
    fps: int = Query(None, ge=1, le=60),
    width: int = Query(960, ge=0, le=3840),
):
    """
    Shared frame preview - reads frames already captured by detection thread.
    Logically decoupled (independent processing) but shares frame source (efficient).
    Instant access, no camera contention, works for webcam AND IP camera.
    """
    boundary = "frame"

    def resize_if_needed(frame):
        if width and frame is not None and frame.shape[1] > width:
            h = int(frame.shape[0] * (width / frame.shape[1]))
            frame = cv2.resize(frame, (width, h), interpolation=cv2.INTER_AREA)
        return frame

    def gen_shared_frame_preview():
        """
        Shared frame preview - reads from self.last_frame (already captured).
        Logically decoupled from detection, shares frame source efficiently.
        
        Benefits:
        - ⚡ Instant access (~1ms vs 30-50ms camera read)
        - ✅ No camera contention (single VideoCapture)
        - ✅ Works with webcam (no dual access)
        - ✅ Works with IP camera (less bandwidth)
        - ✅ Lower latency, better performance
        """
        print(f"🎥 PREVIEW: Starting shared frame preview (instant access, no camera contention)")
        
        # Get configuration
        cfg = STATE.get_config()
        
        # Use config FPS if not provided in query parameter
        actual_fps = fps if fps is not None else cfg.get("runtime", {}).get("frame_read_fps", 20)
        frame_interval = 1.0 / float(actual_fps)
        next_t = time.time()
        
        # If detection is not running, start frame capture for preview
        if not STATE.capture_running:
            print(f"🎥 PREVIEW: Detection not running, starting frame capture for preview")
            STATE._start_frame_capture(cfg)
            time.sleep(0.5)  # Give capture thread time to start
        
        frames_served = 0
        last_log_time = time.time()
        
        try:
            while True:
                # Read latest frame from shared memory (instant!)
                with STATE.lock:
                    if STATE.last_frame is not None:
                        frame = STATE.last_frame.copy()  # ⚡ Fast memory copy (~1ms)
                    else:
                        frame = None
                
                if frame is None:
                    # No frame available yet, wait briefly
                    time.sleep(0.02)
                    continue
                
                # Apply resize if needed
                frame = resize_if_needed(frame)
                
                # Encode to JPEG
                ok, jpeg = cv2.imencode(".jpg", frame, [int(cv2.IMWRITE_JPEG_QUALITY), int(q)])
                if not ok:
                    continue
                
                # Yield MJPEG frame
                yield (b"--" + boundary.encode() + b"\r\n"
                       b"Content-Type: image/jpeg\r\n\r\n" + jpeg.tobytes() + b"\r\n")
                
                frames_served += 1
                
                # Log preview stats every 5 seconds
                if time.time() - last_log_time > 5.0:
                    print(f"🎥 PREVIEW: Served {frames_served} frames in 5s (~{frames_served/5:.1f} FPS)")
                    frames_served = 0
                    last_log_time = time.time()
                
                # FPS throttling
                next_t += frame_interval
                delay = next_t - time.time()
                if delay > 0:
                    time.sleep(delay)
                else:
                    next_t = time.time()
                    
        except Exception as e:
            print(f"❌ PREVIEW: Error - {e}")
        finally:
            print(f"🎥 PREVIEW: Stream ended")
    
    return StreamingResponse(gen_shared_frame_preview(), media_type=f"multipart/x-mixed-replace; boundary={boundary}")

# Enhanced API Endpoints
@app.post("/api/config/save")
async def save_config_to_file():
    """Save current configuration to file with password encoding"""
    try:
        cfg = STATE.get_config()
        
        # Encode password
        if "connection" in cfg and "password" in cfg["connection"]:
            password = cfg["connection"]["password"]
            if password:
                cfg["connection"]["password"] = base64.b64encode(password.encode()).decode()
        
        # Add metadata
        cfg["metadata"] = {
            "last_updated": datetime.now().isoformat(),
            "version": "1.0",
            "description": f"Configuration for gate {cfg.get('gate_id', 'GATE-01')}"
        }
        
        config_file = f"{cfg.get('gate_id', 'GATE-01')}_airport_queue_config.json"
        with open(config_file, 'w') as f:
            json.dump(cfg, f, indent=2)
        
        return {"ok": True, "file": config_file}
    except Exception as e:
        return {"ok": False, "error": str(e)}

@app.get("/api/config/load")
async def load_config_from_file(gate_id: str = "GATE-01"):
    """Load configuration from file with password decoding"""
    try:
        config_file = f"{gate_id}_airport_queue_config.json"
        if os.path.exists(config_file):
            with open(config_file, 'r') as f:
                cfg = json.load(f)
            
            # Decode password
            if "connection" in cfg and "password" in cfg["connection"]:
                encoded_password = cfg["connection"]["password"]
                if encoded_password:
                    try:
                        cfg["connection"]["password"] = base64.b64decode(encoded_password.encode()).decode()
                    except:
                        cfg["connection"]["password"] = ""
            
            return {"ok": True, "config": cfg}
        else:
            return {"ok": False, "error": "config file not found"}
    except Exception as e:
        return {"ok": False, "error": str(e)}

@app.get("/api/history")
def get_detection_history(limit: int = 100):
    """Get detection history"""
    with STATE.lock:
        history = list(STATE.detection_history)[-limit:] if STATE.detection_history else []
        return {
            "history": history,
            "total_detections": STATE.total_detections,
            "average_inference_time": STATE.average_inference_time
        }

@app.delete("/api/history")
def clear_detection_history():
    """Clear detection history"""
    with STATE.lock:
        STATE.detection_history.clear()
        STATE.total_detections = 0
        STATE.average_inference_time = 0.0
    return {"ok": True}

@app.get("/api/models")
def get_available_models():
    """Get list of available detection models"""
    models = {
        'YOLOv8m': {'description': 'Medium YOLO - good speed/accuracy', 'type': 'yolo'},
        'YOLOv8x': {'description': 'Extra-large YOLO - higher accuracy', 'type': 'yolo'},
        'YOLOv9e': {'description': 'YOLOv9 Efficient (GELAN) - strong accuracy', 'type': 'yolo'},
        'YOLOv10x': {'description': 'YOLOv10 X - very high precision', 'type': 'yolo'},
        'YOLOv11l': {'description': 'YOLOv11 L - excellent precision/recall balance', 'type': 'yolo'},
        'YOLOv11x': {'description': 'YOLOv11 X - high accuracy & recall (real-time friendly)', 'type': 'yolo'},
        'YOLOv12l': {'description': 'YOLOv12 L - attention-centric, top recall & mAP', 'type': 'yolo'},
        'YOLOv12x': {'description': 'YOLOv12 X - SOTA-level accuracy (heavier)', 'type': 'yolo'},
        'RT-DETR-X': {'description': 'RT-DETR X - strong for crowded scenes', 'type': 'rtdetr'},
        'DETR-HF': {'description': 'DETR ResNet-50 - end-to-end transformer', 'type': 'detr'}
    }
    
    # Check which models are actually available
    available_models = {}
    try:
        from ultralytics import YOLO
        for model_name, info in models.items():
            if info['type'] in ['yolo', 'rtdetr']:
                available_models[model_name] = info
    except ImportError:
        pass
    
    try:
        from transformers import DetrImageProcessor
        for model_name, info in models.items():
            if info['type'] == 'detr':
                available_models[model_name] = info
    except ImportError:
        pass
    
    return {"models": available_models}

@app.websocket("/ws/status")
async def websocket_status(websocket: WebSocket):
    """WebSocket endpoint for real-time status updates"""
    await websocket.accept()
    try:
        while True:
            cfg = STATE.get_config()
            with STATE.lock:
                data = {
                    "gate_id": cfg.get("gate_id", "GATE-01"),
                    "gate_name": cfg.get("gate_name", "GATE"),
                    "gate_number": cfg.get("gate_number", "01"),
                    "connected": STATE.connected,
                    "running": STATE.running,
                    "people_count": STATE.people_count,
                    "raw_people_count": STATE._raw_people_count,
                    "wait_time": STATE.wait_time,
                    "wait_minutes": STATE.wait_time / 60.0,
                    "wait_seconds": int(STATE.wait_time),
                    "last_frame_ts": STATE.last_frame_ts,
                    "last_error": getattr(STATE, "last_error", None),
                    "total_detections": STATE.total_detections,
                    "average_inference_time": STATE.average_inference_time,
                    "total_cycle_time": getattr(STATE, "total_cycle_time", 0.0),
                    "model": cfg.get("runtime", {}).get("selected_model", "YOLOv11x"),
                    "confidence": cfg.get("runtime", {}).get("confidence_threshold", 0.5),
                    "display_config": cfg.get("display_config", {}),
                    "debug": {
                        "enabled": STATE.debug_enabled,
                        "probability": STATE.debug_probability,
                        "images_saved": STATE.debug_images_saved
                    },
                    "fps": {
                        "actual_fps": STATE.actual_fps,
                        "configured_fps": STATE.configured_fps,
                        "fps_warning": STATE.fps_warning_active,
                        "fps_percentage": round((STATE.actual_fps / STATE.configured_fps * 100), 1) if STATE.configured_fps > 0 else 0
                    },
                    "timestamp": time.time()
                }
            
            await websocket.send_json(data)
            await websocket.receive_text()  # Wait for client ping
    except WebSocketDisconnect:
        pass

@app.get("/healthz")
def healthz():
    return {"ok": True, "status": "running", "connected": STATE.connected}



@app.post("/api/test_connection")
async def test_connection():
    """Test current connection configuration"""
    cfg = STATE.get_config()
    print("🧪 Testing connection...")
    is_connected = STATE.test_connection(cfg)
    conn_type = cfg.get("connection", {}).get("connection_type", "Unknown")
    brand = cfg.get("connection", {}).get("camera_brand", "Generic")
    url = STATE._build_stream_url(cfg)
    
    print(f"🧪 Connection test result: {is_connected}")
    return {
        "connected": is_connected,
        "connection_type": conn_type,
        "camera_brand": brand,
        "url": str(url) if url is not None else "None"
    }

@app.get("/api/diagnose")
def diagnose():
    import numpy as np
    cfg = STATE.get_config()
    with STATE.lock:
        f = STATE.last_frame.copy() if STATE.last_frame is not None else None
        pc = STATE.people_count
        rpc = getattr(STATE, "_raw_people_count", 0)
        err = getattr(STATE, "last_error", None)

    frame_stats = {}
    if f is None:
        frame_stats = {"present": False}
    else:
        nz = int((f > 0).sum())
        total = int(f.size)
        frame_stats = {
            "present": True,
            "shape": list(f.shape),
            "nonzero_ratio": float(nz) / float(total) if total else 0.0,
            "mean": float(f.mean()),
            "max": int(f.max()),
            "min": int(f.min()),
        }

    # Polygon area estimate in normalized space
    poly = (cfg or {}).get("polygon_cropping", {}) or {}
    pts = poly.get("points") or []
    poly_enabled = bool(poly.get("enabled", False))
    poly_points = len(pts)
    
    # Show connection URL for debugging
    connection_url = STATE._build_stream_url(cfg)
    
    return {
        "connected": STATE.connected,
        "running": STATE.running,
        "people_count": pc,
        "raw_people_count": rpc,
        "last_error": err,
        "frame": frame_stats,
        "polygon": {"enabled": poly_enabled, "points": poly_points},
        "runtime": cfg.get("runtime", {}),
        "connection": {
            "type": cfg.get("connection", {}).get("connection_type", "Webcam"),
            "brand": cfg.get("connection", {}).get("camera_brand", "Generic"),
            "url": str(connection_url) if connection_url is not None else "None"
        },
        "model_cache_keys": list(getattr(STATE, "model_cache", {}).keys()),
        "telemetry": {
            "total_detections": STATE.total_detections,
            "average_inference_time": STATE.average_inference_time,
            "history_length": len(STATE.detection_history)
        }
    }
