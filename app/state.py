# app/state.py
import os
import json
import time
import threading
import random
import pickle
import queue
from collections import deque, OrderedDict
from typing import Any, Dict, Optional, List
from datetime import datetime

import cv2
import numpy as np

CONFIG_PATH = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "data", "config.json")
)

# Enhanced Detection Classes
class TinyIOUTracker:
    """IoU-based tracker for maintaining person identities across frames"""
    def __init__(self, iou_thr=0.40, ttl=2):
        self.iou_thr = float(iou_thr)
        self.ttl = int(ttl)
        self.tracks = OrderedDict()
        self.next_id = 1

    @staticmethod
    def _iou(a, b):
        x1 = max(a[0], b[0]); y1 = max(a[1], b[1])
        x2 = min(a[2], b[2]); y2 = min(a[3], b[3])
        iw = max(0.0, x2 - x1); ih = max(0.0, y2 - y1)
        inter = iw * ih
        if inter <= 0: return 0.0
        area_a = (a[2]-a[0])*(a[3]-a[1]); area_b = (b[2]-b[0])*(b[3]-b[1])
        return inter / max(area_a + area_b - inter, 1e-6)

    def update(self, detections):
        dets = [d['bbox'] for d in detections]
        used = set()
        for t in self.tracks.values():
            t['misses'] += 1

        for tid, t in list(self.tracks.items()):
            best_j, best_iou = -1, 0.0
            for j, db in enumerate(dets):
                if j in used: continue
                iou = self._iou(t['bbox'], db)
                if iou > best_iou:
                    best_iou, best_j = iou, j
            if best_iou >= self.iou_thr and best_j >= 0:
                t['bbox'] = dets[best_j]
                t['misses'] = 0
                used.add(best_j)

        for j, db in enumerate(dets):
            if j not in used:
                self.tracks[self.next_id] = {'bbox': db, 'misses': 0}
                self.next_id += 1

        for tid in [tid for tid, t in self.tracks.items() if t['misses'] > self.ttl]:
            del self.tracks[tid]

        return [{'id': tid, **t} for tid, t in self.tracks.items()]

class CountStabilizer:
    """Advanced count stabilization using EMA or Rolling Average"""
    def __init__(self, method="EMA", ema_alpha=0.65, window_frames=3, max_delta=1):
        self.method = method
        self.ema_alpha = ema_alpha
        self.window_frames = window_frames
        self.max_delta = max_delta
        self.ema_value = None
        self.raw_buffer = []
        self.committed_count = 0

    def update(self, raw_count):
        if self.method == "EMA":
            if self.ema_value is None:
                self.ema_value = float(raw_count)
            self.ema_value = self.ema_alpha * float(raw_count) + (1.0 - self.ema_alpha) * float(self.ema_value)
            averaged_count = int(round(self.ema_value))
        else:  # Rolling Average
            self.raw_buffer.append(int(raw_count))
            if len(self.raw_buffer) > self.window_frames:
                self.raw_buffer = self.raw_buffer[-self.window_frames:]
            averaged_count = int(round(sum(self.raw_buffer) / max(1, len(self.raw_buffer))))

        # Rate-limit change per detection tick
        delta = averaged_count - self.committed_count
        if delta > self.max_delta:
            self.committed_count = self.committed_count + self.max_delta
        elif delta < -self.max_delta:
            self.committed_count = self.committed_count - self.max_delta
        else:
            self.committed_count = averaged_count

        return self.committed_count

# Removed QueuePredictor - main app uses simple random calculation


class SharedState:
    def __init__(self):
        self.lock = threading.RLock()
        self.config = self._load_config()  # type: Dict[str, Any]

        # Live stats
        self.people_count = 0               # stabilized + adjusted
        self._raw_people_count = 0          # raw detector output
        self._ema_value = None              # type: Optional[float]
        self._rolling = deque(maxlen=10)    # rolling avg buffer

        self.wait_time = 0.0                # seconds
        self.last_frame = None              # BGR frame (already cropped/masked)
        self.last_frame_ts = 0.0
        
        # Enhanced features
        self.tracker = None                 # IoU tracker instance
        self.stabilizer = None              # Count stabilizer instance
        self.detection_history = deque(maxlen=200)  # Detection history
        self.total_detections = 0
        self.average_inference_time = 0.0

        # Runtime states
        self.connected = False
        self.stop_flag = False
        self.running = False
        self.thread = None                  # type: Optional[threading.Thread]
        
        # Frame capture system (like legacy QMS)
        self.frame_queue = queue.Queue(maxsize=2)  # Keep only latest frames
        self.capture_thread = None          # type: Optional[threading.Thread]
        self.capture_running = False
        self.current_stream_url = None

        # Models + diagnostics
        self.model_cache = {}               # type: Dict[str, Any]
        self.last_error = None              # type: Optional[str]
        self._current_capture = None        # type: Optional[cv2.VideoCapture]

    # ---------------- config ----------------

    def _load_config(self):
        try:
            with open(CONFIG_PATH, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            return {}

    def save_config(self, cfg):
        try:
            with open(CONFIG_PATH, "w", encoding="utf-8") as f:
                json.dump(cfg, f, indent=2)
            with self.lock:
                self.config = cfg
            return True
        except Exception as e:
            print("Failed to save config:", e)
            return False

    def get_config(self):
        with self.lock:
            return self.config

    def test_connection(self, cfg=None):
        """Test connection without starting detection"""
        if cfg is None:
            cfg = self.get_config()
        
        try:
            cap = self._open_capture(cfg)
            if cap is None:
                return False
            
            is_connected = cap.isOpened()
            if is_connected:
                # Try to read one frame to verify
                try:
                    ok, frame = cap.read()
                    is_connected = ok and frame is not None
                except:
                    is_connected = False
            
            try:
                cap.release()
            except:
                pass
                
            return is_connected
        except Exception as e:
            print(f"Connection test failed: {e}")
            return False

    # ---------------- camera ----------------

    def _build_stream_url(self, cfg):
        """
        Build stream URL based on connection type and camera brand.
        Based on legacy QMS logic with full brand support.
        """
        conn = (cfg or {}).get("connection", {}) or {}
        connection_type = (conn.get("connection_type") or "Webcam").strip()
        
        if connection_type == "Custom URL":
            return conn.get("custom_url") or ""
        elif connection_type == "Webcam":
            return int(conn.get("webcam_index", 0))
        
        # Extract connection details
        ip_address = conn.get("ip_address", "") or ""
        port = int(conn.get("port", 554))
        username = conn.get("username", "") or ""
        password = conn.get("password", "") or ""
        channel = int(conn.get("channel", 1))
        stream_quality = (conn.get("stream_quality") or "sub").strip()  # "sub"|"main"
        camera_brand = (conn.get("camera_brand") or "Generic").strip()
        
        if connection_type == "NVR":
            if stream_quality == "main":
                return f"rtsp://{username}:{password}@{ip_address}:{port}/cam/realmonitor?channel={channel}&subtype=0"
            else:
                return f"rtsp://{username}:{password}@{ip_address}:{port}/cam/realmonitor?channel={channel}&subtype=1"
                
        elif connection_type == "Direct Camera":
            if camera_brand == "Pelco":
                return f"rtsp://{username}:{password}@{ip_address}:{port}/rtsp/defaultPrimary?streamType=u" if stream_quality=="main" else f"rtsp://{username}:{password}@{ip_address}:{port}/rtsp/defaultSecondary?streamType=u"
            elif camera_brand == "Hikvision":
                return f"rtsp://{username}:{password}@{ip_address}:{port}/Streaming/Channels/101/httppreview" if stream_quality=="main" else f"rtsp://{username}:{password}@{ip_address}:{port}/Streaming/Channels/102/httppreview"
            elif camera_brand == "Dahua":
                return f"rtsp://{username}:{password}@{ip_address}:{port}/cam/realmonitor?channel=1&subtype=0" if stream_quality=="main" else f"rtsp://{username}:{password}@{ip_address}:{port}/cam/realmonitor?channel=1&subtype=1"
            elif camera_brand == "Axis":
                return f"rtsp://{username}:{password}@{ip_address}:{port}/axis-media/media.amp?videocodec=h264" if stream_quality=="main" else f"rtsp://{username}:{password}@{ip_address}:{port}/axis-media/media.amp?resolution=320x240"
            else:  # Generic
                return f"rtsp://{username}:{password}@{ip_address}:{port}/stream1" if stream_quality=="main" else f"rtsp://{username}:{password}@{ip_address}:{port}/stream2"
                
        elif connection_type == "HTTP Stream":
            if camera_brand == "Pelco":
                return f"http://{username}:{password}@{ip_address}:{port}/media/cam0/still.jpg"
            elif camera_brand == "Hikvision":
                return f"http://{username}:{password}@{ip_address}:{port}/ISAPI/Streaming/channels/101/picture"
            elif camera_brand == "Dahua":
                return f"http://{username}:{password}@{ip_address}:{port}/cgi-bin/mjpg/video.cgi"
            elif camera_brand == "Axis":
                return f"http://{username}:{password}@{ip_address}:{port}/axis-cgi/mjpg/video.cgi"
            else:  # Generic
                return f"http://{username}:{password}@{ip_address}:{port}/video.cgi"
        
        # Fallback for URL type
        return conn.get("custom_url") or ""

    def _open_capture(self, cfg):
        """
        OpenCV VideoCapture with proper connection type handling.
        Supports Webcam, URL, NVR, Direct Camera, HTTP Stream.
        """
        # Release any existing capture first
        if hasattr(self, '_current_capture') and self._current_capture is not None:
            try:
                self._current_capture.release()
            except Exception:
                pass
            self._current_capture = None

        stream_url = self._build_stream_url(cfg)
        
        if isinstance(stream_url, int):
            # Webcam index
            cap = cv2.VideoCapture(stream_url)
        elif isinstance(stream_url, str) and stream_url:
            # URL string (RTSP, HTTP, etc.)
            cap = cv2.VideoCapture(stream_url)
        else:
            # Fallback to webcam 0
            cap = cv2.VideoCapture(0)
            
        # Store current capture for proper cleanup
        self._current_capture = cap
        return cap

    # ---------------- cropping ----------------

    def _apply_crop(self, frame, cfg):
        """
        1) Rect crop (percentages in runtime.*)
        2) Polygon mask (normalized points in polygon_cropping.points)
        """
        rt = (cfg or {}).get("runtime", {}) or {}
        poly = (cfg or {}).get("polygon_cropping", {}) or {}

        # Rectangular crop
        if bool(rt.get("enable_cropping", False)):
            h, w = frame.shape[:2]
            l = max(0, min(100, int(rt.get("crop_left", 0)))) / 100.0
            t = max(0, min(100, int(rt.get("crop_top", 0)))) / 100.0
            r = max(0, min(100, int(rt.get("crop_right", 100)))) / 100.0
            b = max(0, min(100, int(rt.get("crop_bottom", 100)))) / 100.0
            x1, y1, x2, y2 = int(w * l), int(h * t), int(w * r), int(h * b)
            x1 = max(0, min(w - 1, x1))
            x2 = max(0, min(w, x2))
            y1 = max(0, min(h - 1, y1))
            y2 = max(0, min(h, y2))
            if x2 > x1 and y2 > y1:
                frame = frame[y1:y2, x1:x2]

        # Polygon mask
        if bool(poly.get("enabled", False)):
            pts = poly.get("points") or []
            if isinstance(pts, list) and len(pts) >= 3:
                hh, ww = frame.shape[:2]
                arr = []
                for p in pts:
                    try:
                        px = int(max(0.0, min(1.0, float(p["x"]))) * ww)
                        py = int(max(0.0, min(1.0, float(p["y"]))) * hh)
                        arr.append([px, py])
                    except Exception:
                        pass
                if len(arr) >= 3:
                    mask = np.zeros((hh, ww), dtype=np.uint8)
                    cv2.fillPoly(mask, [np.array(arr, dtype=np.int32)], 255)
                    frame = cv2.bitwise_and(frame, frame, mask=mask)

        return frame

    # ---------------- models ----------------

    def _infer_ultralytics(self, frame, cfg):
        """
        Enhanced Ultralytics path with advanced detection features.
        Returns detailed detection results with enhanced filtering.
        """
        from ultralytics import YOLO, RTDETR
        import torch

        runtime = (cfg or {}).get("runtime", {}) or {}
        model_name = str(runtime.get("selected_model", "YOLOv11x"))
        conf = float(runtime.get("confidence_threshold", 0.5))
        imgsz = int(runtime.get("imgsz", 1280))
        half = bool(runtime.get("half_precision", False))
        use_tta = bool(runtime.get("use_tta", False))
        max_det = int(runtime.get("max_det", 100))
        secondary_iou = float(runtime.get("secondary_iou", 0.70))
        min_height_ratio = float(runtime.get("min_height_ratio", 0.018))
        min_area_ratio = float(runtime.get("min_area_ratio", 0.00015))

        # Enhanced model mapping
        model_files = {
            'YOLOv8m': 'yolov8m.pt',
            'YOLOv8x': 'yolov8x.pt', 
            'YOLOv9e': 'yolov9e.pt',
            'YOLOv10x': 'yolov10x.pt',
            'YOLOv11l': 'yolo11l.pt',
            'YOLOv11x': 'yolo11x.pt',
            'YOLOv12l': 'yolo12l.pt',
            'YOLOv12x': 'yolo12x.pt',
            'RT-DETR-X': 'rtdetr-x.pt'
        }
        
        model_file = model_files.get(model_name, f"{model_name}.pt")
        key = f"ultra::{model_file}::{half}"
        
        if key not in self.model_cache:
            try:
                if "RT-DETR" in model_name.upper():
                    self.model_cache[key] = RTDETR(model_file)
                else:
                    self.model_cache[key] = YOLO(model_file)
            except Exception:
                self.model_cache[key] = YOLO("yolov8m.pt")  # fallback
                
        model = self.model_cache[key]

        if frame is None or frame.size == 0:
            return {'people_count': 0, 'detections': [], 'inference_time': 0.0}

        start_time = time.time()
        
        # Enhanced detection with TTA and max_det
        with torch.inference_mode():
            results = model.predict(
                source=frame, 
                imgsz=imgsz, 
                conf=conf, 
                half=half, 
                verbose=False,
                classes=[0],  # person only
                max_det=max_det,
                augment=use_tta
            )
        
        inference_time = time.time() - start_time
        h, w = frame.shape[:2]
        img_area = float(max(1, h * w))
        min_h = max(1.0, h * min_height_ratio)
        min_area = max(1.0, img_area * min_area_ratio)
        
        # Get polygon ROI if enabled
        poly = (cfg or {}).get("polygon_cropping", {}) or {}
        roi_polygon = None
        if bool(poly.get("enabled", False)):
            pts = poly.get("points") or []
            if len(pts) >= 3:
                roi_polygon = [(int(p["x"]/100.0*w), int(p["y"]/100.0*h)) for p in pts]
        
        def _inside_roi_center(bbox, polygon):
            if not polygon:
                return True
            cx = int((bbox[0] + bbox[2]) * 0.5)
            cy = int((bbox[1] + bbox[3]) * 0.5)
            poly = np.array(polygon, dtype=np.int32).reshape(-1, 1, 2)
            return cv2.pointPolygonTest(poly, (cx, cy), False) >= 0
        
        detections = []
        for r in results:
            for b in getattr(r, 'boxes', []) or []:
                xyxy = b.xyxy[0].tolist()
                score = float(b.conf[0].item())
                x1, y1, x2, y2 = map(float, xyxy)
                
                # Apply enhanced filters
                if (y2 - y1) < min_h:
                    continue
                if (x2 - x1) * (y2 - y1) < min_area:
                    continue
                if not _inside_roi_center([x1, y1, x2, y2], roi_polygon):
                    continue
                    
                detections.append({
                    'bbox': [int(x1), int(y1), int(x2), int(y2)],
                    'confidence': score
                })
        
        # Secondary IoU suppression
        def _iou(a, b):
            ax1, ay1, ax2, ay2 = a['bbox']
            bx1, by1, bx2, by2 = b['bbox']
            inter_x1 = max(ax1, bx1)
            inter_y1 = max(ay1, by1)
            inter_x2 = min(ax2, bx2)
            inter_y2 = min(ay2, by2)
            inter = max(0, inter_x2 - inter_x1) * max(0, inter_y2 - inter_y1)
            area_a = max(0, ax2-ax1) * max(0, ay2-ay1)
            area_b = max(0, bx2-bx1) * max(0, by2-by1)
            union = area_a + area_b - inter + 1e-6
            return inter / union
        
        detections.sort(key=lambda d: d['confidence'], reverse=True)
        kept = []
        for cand in detections:
            if all(_iou(cand, k) < secondary_iou for k in kept):
                kept.append(cand)
        
        return {
            'people_count': len(kept),
            'detections': kept,
            'inference_time': inference_time,
            'model_name': model_name
        }

    def _infer_transformers_detr(self, frame, cfg):
        """
        HuggingFace Transformers DETR (facebook/detr-resnet-50).
        Returns number of 'person' label (id=1 in post-processed results).
        """
        import torch
        from transformers import DetrImageProcessor, DetrForObjectDetection

        runtime = (cfg or {}).get("runtime", {}) or {}
        conf = float(runtime.get("confidence_threshold", 0.5))

        key = "transformers::detr-resnet-50"
        if key not in self.model_cache:
            proc = DetrImageProcessor.from_pretrained("facebook/detr-resnet-50")
            det = DetrForObjectDetection.from_pretrained("facebook/detr-resnet-50")
            self.model_cache[key] = (proc, det)

        processor, model = self.model_cache[key]

        # BGR -> RGB
        rgb = frame[:, :, ::-1]
        inputs = processor(images=rgb, return_tensors="pt")
        with torch.no_grad():
            outputs = model(**inputs)

        target_sizes = torch.tensor([rgb.shape[:2]])
        results = processor.post_process_object_detection(
            outputs, target_sizes=target_sizes, threshold=conf
        )[0]

        people = 0
        for score, label in zip(results["scores"], results["labels"]):
            try:
                if int(label.item()) == 1:  # COCO person
                    people += 1
            except Exception:
                pass

        return people

    def _infer(self, frame, cfg):
        """
        Route to model family; capture errors for diagnostics.
        """
        runtime = (cfg or {}).get("runtime", {}) or {}
        model_name = str(runtime.get("selected_model", "YOLOv11x")).lower()
        try:
            if ("transformers" in model_name) and ("detr" in model_name):
                self.last_error = None
                return self._infer_transformers_detr(frame, cfg)
            self.last_error = None
            return self._infer_ultralytics(frame, cfg)
        except Exception as e:
            self.last_error = "{}: {}".format(type(e).__name__, e)
            # heuristic fallback to avoid crashing UI
            return int(np.mean(frame) / 32)

    # ---------------- stabilization & wait-time ----------------

    def _stabilize(self, raw_count, cfg):
        """
        Enhanced stabilization using dedicated CountStabilizer class.
        """
        cs = (cfg or {}).get("count_stabilization", {}) or {}
        method = (cs.get("method") or "EMA")
        avg_window = int(cs.get("avg_window_frames", 3))
        ema_alpha = float(cs.get("ema_alpha", 0.65))
        max_delta = int(cs.get("max_delta_per_detection", 1))
        
        # Initialize or update stabilizer if config changed
        stabilizer_cfg = (method, ema_alpha, avg_window, max_delta)
        if not hasattr(self, '_stabilizer_cfg') or self._stabilizer_cfg != stabilizer_cfg:
            self.stabilizer = CountStabilizer(
                method=method,
                ema_alpha=ema_alpha,
                window_frames=avg_window,
                max_delta=max_delta
            )
            self._stabilizer_cfg = stabilizer_cfg
        
        return self.stabilizer.update(raw_count)

    def _estimate_wait(self, count, cfg):
        """
        Simple wait time estimation using random per-person time (matches main app exactly).
        """
        rt = (cfg or {}).get("runtime", {}) or {}
        tmin = float(rt.get("per_person_time_min", 22))
        tmax = float(rt.get("per_person_time_max", 26))
        
        if count > 0:
            if tmax < tmin:
                tmax = tmin
            per_person_seconds = random.randint(int(tmin), int(tmax))
            return count * per_person_seconds
        return 0.0

    # ---------------- frame capture system (like legacy QMS) ----------------
    
    def _start_frame_capture(self, cfg):
        """Start dedicated frame capture thread"""
        self._stop_frame_capture()
        
        stream_url = self._build_stream_url(cfg)
        if stream_url == self.current_stream_url and self.capture_running:
            return  # Already running with same config
            
        self.current_stream_url = stream_url
        self.capture_running = True
        self.capture_thread = threading.Thread(target=self._capture_frames, daemon=True)
        self.capture_thread.start()
        print(f"🎬 Started frame capture thread for {cfg.get('connection', {}).get('connection_type', 'Unknown')}")
    
    def _stop_frame_capture(self):
        """Stop frame capture thread"""
        self.capture_running = False
        if self.capture_thread and self.capture_thread.is_alive():
            self.capture_thread.join(timeout=2.0)
        
        # Clear frame queue
        while not self.frame_queue.empty():
            try:
                self.frame_queue.get_nowait()
            except queue.Empty:
                break
                
        # Release capture
        if hasattr(self, '_frame_capture') and self._frame_capture is not None:
            try:
                self._frame_capture.release()
            except:
                pass
            self._frame_capture = None
            
        print(f"🛑 Stopped frame capture thread")
    
    def _capture_frames(self):
        """Dedicated frame capture thread (like legacy QMS)"""
        cap = None
        try:
            # Open capture
            if isinstance(self.current_stream_url, int):
                cap = cv2.VideoCapture(self.current_stream_url)
            elif isinstance(self.current_stream_url, str) and self.current_stream_url:
                cap = cv2.VideoCapture(self.current_stream_url)
            else:
                cap = cv2.VideoCapture(0)
                
            if cap is not None:
                cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
                try:
                    cap.set(cv2.CAP_PROP_TIMEOUT, 10000)
                except:
                    pass
                    
            self._frame_capture = cap
            self.connected = cap is not None and cap.isOpened()
            
            # Frame capture loop
            target_fps = 30  # High FPS for smooth preview
            frame_interval = 1.0 / target_fps
            
            while self.capture_running and cap and cap.isOpened():
                try:
                    ret, frame = cap.read()
                    if ret and frame is not None:
                        # Apply cropping
                        cfg = self.get_config()
                        frame = self._apply_crop(frame, cfg)
                        
                        # Update frame queue (keep only latest)
                        try:
                            if not self.frame_queue.empty():
                                try:
                                    self.frame_queue.get_nowait()
                                except queue.Empty:
                                    pass
                            self.frame_queue.put(frame, timeout=0.01)
                            
                            # Also update last_frame for compatibility
                            with self.lock:
                                self.last_frame = frame.copy()
                                self.last_frame_ts = time.time()
                                
                        except queue.Full:
                            pass
                    else:
                        # Reconnect on read failure
                        print(f"🔄 Frame read failed, attempting reconnect...")
                        time.sleep(1.0)
                        if self.capture_running and self.current_stream_url:
                            try:
                                cap.release()
                                if isinstance(self.current_stream_url, int):
                                    cap = cv2.VideoCapture(self.current_stream_url)
                                else:
                                    cap = cv2.VideoCapture(self.current_stream_url)
                                cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
                                self._frame_capture = cap
                                self.connected = cap.isOpened()
                            except:
                                break
                except Exception as e:
                    print(f"❌ Frame capture error: {e}")
                    break
                    
                time.sleep(max(0.0, frame_interval))
                
        finally:
            if cap is not None:
                try:
                    cap.release()
                except:
                    pass
            self.connected = False
            print(f"🎬 Frame capture thread ended")
    
    def get_latest_frame(self):
        """Get latest frame from queue (non-blocking like legacy QMS)"""
        try:
            return self.frame_queue.get_nowait()
        except queue.Empty:
            return None

    # ---------------- worker loop ----------------

    def start(self):
        if self.thread and self.thread.is_alive():
            self.running = True
            print("🚀 Detection already running, resuming...")
            return
        print("🚀 Starting detection system...")
        
        # Start frame capture thread first
        cfg = self.get_config()
        self._start_frame_capture(cfg)
        
        # Then start detection thread
        self.stop_flag = False
        self.running = True
        self.thread = threading.Thread(target=self._loop, daemon=True)
        self.thread.start()

    def stop(self):
        self.stop_flag = True
        self.running = False
        
        # Stop detection thread
        if self.thread:
            self.thread.join(timeout=1.0)
            
        # Stop frame capture thread
        self._stop_frame_capture()
        
        print("🛑 Detection system stopped")

    def _loop(self):
        """Detection loop that uses frames from the queue system"""
        last_det = 0.0
        frame_count = 0
        
        while not self.stop_flag and self.running:
            cfg = self.get_config()
            
            # Check if connection config changed - restart frame capture if needed
            new_connection_config = (cfg or {}).get("connection", {})
            new_stream_url = self._build_stream_url(cfg)
            if new_stream_url != self.current_stream_url:
                print(f"🔄 DETECTION: Connection config changed, restarting frame capture...")
                self._start_frame_capture(cfg)

            # Get frame from queue (non-blocking)
            frame = self.get_latest_frame()
            if frame is None:
                time.sleep(0.1)  # Wait for frames
                continue
                
            frame_count += 1
            now = time.time()
            
            if frame_count % 30 == 0:
                print(f"📹 Detection frame #{frame_count}: {frame.shape}")

            # Clamp detection interval 0.1..10.0
            det_interval = float((cfg or {}).get("runtime", {}).get("detection_interval", 1.0))
            det_interval = max(0.1, min(10.0, det_interval))
            
            time_since_last = now - last_det
            if frame_count % 60 == 0:  # Log every 60 frames
                print(f"⏱️  Detection timing: {time_since_last:.2f}s since last (interval: {det_interval}s)")

            if time_since_last >= det_interval:
                print(f"🤖 Starting detection inference (interval: {det_interval}s)")
                detection_results = self._infer(frame, cfg)
                print(f"🔍 Detection results: {detection_results if isinstance(detection_results, dict) else f'count={detection_results}'}")
                
                if isinstance(detection_results, dict):
                    raw_pc = detection_results.get('people_count', 0)
                    detections = detection_results.get('detections', [])
                    inference_time = detection_results.get('inference_time', 0.0)
                    model_name = detection_results.get('model_name', 'Unknown')
                else:
                    raw_pc = detection_results
                    detections = []
                    inference_time = 0.0
                    model_name = 'Legacy'
                
                # Apply tracking if enabled
                tracker_cfg = (cfg or {}).get("tracker", {}) or {}
                if bool(tracker_cfg.get("enabled", False)):
                    if not hasattr(self, '_tracker_cfg') or self._tracker_cfg != tracker_cfg:
                        self.tracker = TinyIOUTracker(
                            ttl=int(tracker_cfg.get("ttl", 2)),
                            iou_thr=float(tracker_cfg.get("iou_thr", 0.40))
                        )
                        self._tracker_cfg = tracker_cfg
                    
                    tracks = self.tracker.update(detections)
                    tracked_count = sum(1 for t in tracks if t.get('misses', 0) <= 1)
                    raw_pc = tracked_count
                
                # Stabilize count
                st_pc = self._stabilize(raw_pc, cfg)

                # Apply people_adjustment after stabilization
                adj = int((cfg or {}).get("runtime", {}).get("people_adjustment", -1))
                final_count = max(0, st_pc + adj)
                
                # Calculate wait time
                wait_seconds = self._estimate_wait(final_count, cfg)

                with self.lock:
                    self._raw_people_count = raw_pc
                    self.people_count = final_count
                    self.wait_time = wait_seconds
                    
                    print(f"📊 DETECTION UPDATE:")
                    print(f"   Raw count: {raw_pc}")
                    print(f"   Stabilized: {st_pc}")
                    print(f"   Final count: {final_count}")
                    print(f"   Wait time: {wait_seconds:.1f}s")
                    
                    # Update telemetry
                    self.total_detections += 1
                    print(f"   Total detections: {self.total_detections}")
                    if self.total_detections > 0:
                        if len(self.detection_history) > 0:
                            recent_times = [r['inference_time'] for r in list(self.detection_history)[-10:]]
                            self.average_inference_time = sum(recent_times) / len(recent_times)
                        else:
                            self.average_inference_time = inference_time
                    
                    # Add to detection history
                    detection_record = {
                        'timestamp': now,
                        'people_count': final_count,
                        'wait_time': wait_seconds / 60.0,  # minutes
                        'model_name': model_name,
                        'confidence': float((cfg or {}).get("runtime", {}).get("confidence_threshold", 0.5)),
                        'inference_time': inference_time
                    }
                    self.detection_history.append(detection_record)

                last_det = now

            # Fast detection loop - don't slow it down
            time.sleep(0.1)


STATE = SharedState()
