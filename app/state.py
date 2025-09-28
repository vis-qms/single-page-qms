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
from ultralytics import YOLO
import torch

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
    """Advanced count stabilization using EMA, Rolling Average, or 3-Frame Average (Median)"""
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
        elif self.method == "Median":
            # Median stabilization - averaging is done at detection level (3 consecutive frames)
            # Just pass through the already-averaged result
            averaged_count = int(raw_count)
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
        self.last_frame = None              # BGR frame (display with overlay)
        self.last_detection_frame = None    # BGR frame (cropped for detection)
        self.last_frame_ts = 0.0
        
        # Wait time caching to avoid recalculation when people count is same
        self._last_wait_people_count = -1   # Track last people count for wait time
        self._cached_wait_time = 0.0        # Cached wait time for same people count
        
        # Debug image saving
        self.debug_enabled = False          # Enable/disable debug image saving
        self.debug_probability = 10         # Probability (0-100) to save debug images
        self.debug_images_saved = 0         # Counter for saved debug images
        
        print(f"🐛 DEBUG INIT: enabled={self.debug_enabled}, probability={self.debug_probability}%")
        
        # Enhanced features
        self.tracker = None                 # IoU tracker instance
        self.stabilizer = None              # Count stabilizer instance
        self.detection_history = deque(maxlen=200)  # Detection history
        self.total_detections = 0
        self.average_inference_time = 0.0
        self.total_cycle_time = 0.0         # Total detection cycle time

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
        
        # Consecutive frame buffer for batch inference
        self.consecutive_frame_buffer = deque(maxlen=10)  # Store last 10 consecutive frames
        self.frame_buffer_lock = threading.Lock()  # Thread-safe access to frame buffer

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
                
            # Debug polygon config when saving
            poly = cfg.get("polygon_cropping", {})
            if poly.get("enabled"):
                pts = poly.get("points", [])
                print(f"🔧 Config saved - Polygon enabled with {len(pts)} points:")
                for i, p in enumerate(pts[:3]):  # Show first 3 points
                    print(f"   Point {i+1}: x={p['x']:.3f}, y={p['y']:.3f}")
                if len(pts) > 3:
                    print(f"   ... and {len(pts)-3} more points")
            else:
                print("🔧 Config saved - Polygon cropping disabled")
                
            return True
        except Exception as e:
            print("Failed to save config:", e)
            return False

    def get_config(self):
        with self.lock:
            # Sync debug settings from config
            debug_cfg = self.config.get("debug", {})
            old_enabled = self.debug_enabled
            old_prob = self.debug_probability
            self.debug_enabled = bool(debug_cfg.get("enabled", False))
            self.debug_probability = int(debug_cfg.get("probability", 10))
            
            if old_enabled != self.debug_enabled or old_prob != self.debug_probability:
                print(f"🐛 DEBUG CONFIG: enabled={self.debug_enabled}, probability={self.debug_probability}%")
            
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
    
    def _polygon_points_to_pixels(self, polygon_points, w, h):
        """Convert normalized polygon points to pixel coordinates - IDENTICAL to frontend polygon editor"""
        if not polygon_points:
            return []
        
        pixel_points = []
        for p in polygon_points:
            try:
                # Use proper rounding to match frontend exactly
                # Frontend uses: pt.x * rect.width, pt.y * rect.height
                x_raw = float(p["x"]) * w
                y_raw = float(p["y"]) * h
                x = round(x_raw)
                y = round(y_raw)
                # Don't clamp to w-1, h-1 - allow full range like frontend
                x = max(0, min(w, x))
                y = max(0, min(h, y))
                pixel_points.append([x, y])
                
            except Exception as e:
                print(f"❌ Error parsing polygon point {p}: {e}")
                continue
        return pixel_points

    def _apply_crop_for_detection(self, frame, cfg):
        """
        Apply cropping for AI detection exactly like legacy QMS.
        This CROPS the frame size (like PIL crop), not just masks it.
        """
        rt = (cfg or {}).get("runtime", {}) or {}
        poly = (cfg or {}).get("polygon_cropping", {}) or {}

        # Rectangular crop (like legacy crop_image function)
        if bool(rt.get("enable_cropping", False)):
            h, w = frame.shape[:2]
            l = max(0, min(100, int(rt.get("crop_left", 0)))) / 100.0
            t = max(0, min(100, int(rt.get("crop_top", 0)))) / 100.0
            r = max(0, min(100, int(rt.get("crop_right", 100)))) / 100.0
            b = max(0, min(100, int(rt.get("crop_bottom", 100)))) / 100.0
            
            # Convert to pixel coordinates with proper rounding
            left = round(w * l)
            top = round(h * t) 
            right = round(w * r)
            bottom = round(h * b)
            
            # Clamp to valid bounds
            left = max(0, min(left, w))
            top = max(0, min(top, h))
            right = max(left, min(right, w))
            bottom = max(top, min(bottom, h))
            
            # Actually crop the frame (like PIL crop)
            if right > left and bottom > top:
                frame = frame[top:bottom, left:right]
                print(f"🔲 Rectangle crop applied: ({left},{top}) to ({right},{bottom}) -> {frame.shape}")

        # Polygon mask (like legacy apply_polygon_mask_pil - keeps same frame size)
        # Remove the rectangle cropping check - polygon should work independently
        if bool(poly.get("enabled", False)):
            pts = poly.get("points") or []
            if isinstance(pts, list) and len(pts) >= 3:
                h, w = frame.shape[:2]
                # Use IDENTICAL coordinate conversion as frontend polygon editor
                pixel_points = self._polygon_points_to_pixels(pts, w, h)
                if len(pixel_points) >= 3:
                    # Create mask and apply (keeps same frame dimensions)
                    mask = np.zeros((h, w), dtype=np.uint8)
                    cv2.fillPoly(mask, [np.array(pixel_points, dtype=np.int32)], 255)
                    frame = cv2.bitwise_and(frame, frame, mask=mask)

        return frame

    def _apply_display_overlay(self, frame, cfg):
        """
        Apply visual overlay for preview display (like legacy QMS).
        Shows full frame with dimmed areas outside crop/polygon.
        NO GREEN LINES - just dimming effect.
        """
        rt = (cfg or {}).get("runtime", {}) or {}
        poly = (cfg or {}).get("polygon_cropping", {}) or {}
        
        display_frame = frame.copy()
        
        # Rectangle cropping overlay - just dim outside, no green lines
        if bool(rt.get("enable_cropping", False)):
            h, w = frame.shape[:2]
            l = max(0, min(100, int(rt.get("crop_left", 0)))) / 100.0
            t = max(0, min(100, int(rt.get("crop_top", 0)))) / 100.0
            r = max(0, min(100, int(rt.get("crop_right", 100)))) / 100.0
            b = max(0, min(100, int(rt.get("crop_bottom", 100)))) / 100.0
            x1, y1, x2, y2 = round(w * l), round(h * t), round(w * r), round(h * b)
            x1 = max(0, min(w - 1, x1))
            x2 = max(0, min(w, x2))
            y1 = max(0, min(h - 1, y1))
            y2 = max(0, min(h, y2))
            
            # Dim outside area only
            overlay = (display_frame * 0.5).astype(np.uint8)
            display_frame = overlay.copy()
            if x2 > x1 and y2 > y1:
                display_frame[y1:y2, x1:x2] = frame[y1:y2, x1:x2]  # Keep original inside
        
        # Polygon cropping overlay (like legacy draw_polygon_overlay_bgr) - no green lines
        if bool(poly.get("enabled", False)):
            pts = poly.get("points") or []
            if isinstance(pts, list) and len(pts) >= 3:
                h, w = frame.shape[:2]
                # Use IDENTICAL coordinate conversion as frontend polygon editor
                pixel_points = self._polygon_points_to_pixels(pts, w, h)
                if len(pixel_points) >= 3:
                    poly_pts = np.array(pixel_points, dtype=np.int32)
                    
                    # Create overlay with red polygon outline (exactly like fastapi-qms_copy)
                    overlay = display_frame.copy()
                    cv2.polylines(overlay, [poly_pts], isClosed=True, color=(0, 0, 255), thickness=2)
                    
                    # Create mask for polygon area (identical to detection mask)
                    mask = np.zeros((h, w), dtype=np.uint8)
                    cv2.fillPoly(mask, [poly_pts], 255)
                    
                    # Dim everything to 30% (like fastapi-qms_copy)
                    dim = (display_frame * 0.3).astype(display_frame.dtype)
                    
                    # Show overlay (with red outline) inside polygon, dimmed outside
                    display_frame = np.where(mask[..., None] == 255, overlay, dim)
        
        return display_frame

    # ---------------- models ----------------

    def _get_model_params(self, cfg):
        """Extract common model parameters to avoid duplication"""
        runtime = (cfg or {}).get("runtime", {}) or {}
        return {
            'conf': float(runtime.get("confidence_threshold", 0.5)),
            'imgsz': int(runtime.get("imgsz", 1280)),
            'half': bool(runtime.get("half_precision", False)),
            'use_tta': bool(runtime.get("use_tta", False)),
            'max_det': int(runtime.get("max_det", 100)),
            'primary_iou': float(runtime.get("primary_iou", 0.68)),
            'secondary_iou': float(runtime.get("secondary_iou", 0.70)),
            'min_height_ratio': float(runtime.get("min_height_ratio", 0.018)),
            'min_area_ratio': float(runtime.get("min_area_ratio", 0.00015))
        }

    def _infer_ultralytics(self, frame, cfg):
        """
        Enhanced Ultralytics path with advanced detection features.
        Returns detailed detection results with enhanced filtering.
        """
        runtime = (cfg or {}).get("runtime", {}) or {}
        model_name = str(runtime.get("selected_model", "YOLOv11x"))
        params = self._get_model_params(cfg)

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
        }
        
        model_file = model_files.get(model_name, f"{model_name}.pt")
        key = f"ultra::{model_file}::{params['half']}"
        
        if key not in self.model_cache:
            try:
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
                imgsz=params['imgsz'], 
                conf=params['conf'], 
                iou=params['primary_iou'],  # Primary IoU for model's NMS
                half=params['half'], 
                verbose=False,
                classes=[0],  # person only
                max_det=params['max_det'],
                augment=params['use_tta']
            )
        
        inference_time = time.time() - start_time
        h, w = frame.shape[:2]
        img_area = float(max(1, h * w))
        min_h = max(1.0, h * params['min_height_ratio'])
        min_area = max(1.0, img_area * params['min_area_ratio'])
        
        # Get polygon ROI if enabled
        poly = (cfg or {}).get("polygon_cropping", {}) or {}
        roi_polygon = None
        if bool(poly.get("enabled", False)):
            pts = poly.get("points") or []
            if len(pts) >= 3:
                # Use IDENTICAL coordinate conversion as frontend polygon editor
                pixel_points = self._polygon_points_to_pixels(pts, w, h)
                roi_polygon = [(pt[0], pt[1]) for pt in pixel_points]
        
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
            if all(_iou(cand, k) < params['secondary_iou'] for k in kept):
                kept.append(cand)
        
        return {
            'people_count': len(kept),
            'detections': kept,
            'inference_time': inference_time,
            'model_name': model_name
        }


    def _infer(self, frame, cfg):
        """
        Route to model family; capture errors for diagnostics.
        """
        runtime = (cfg or {}).get("runtime", {}) or {}
        model_name = str(runtime.get("selected_model", "YOLOv11x")).lower()
        try:
            self.last_error = None
            return self._infer_ultralytics(frame, cfg)
        except Exception as e:
            self.last_error = "{}: {}".format(type(e).__name__, e)
            # heuristic fallback to avoid crashing UI
            return int(np.mean(frame) / 32)

    def _infer_batch(self, frame_list, cfg):
        """
        Batch inference for multiple consecutive frames.
        Much more efficient than individual frame inference.
        """
        if not frame_list:
            return []
            
        runtime = (cfg or {}).get("runtime", {}) or {}
        model_name = str(runtime.get("selected_model", "YOLOv11x"))
        params = self._get_model_params(cfg)
        
        print(f"🚀 BATCH INFERENCE: Processing {len(frame_list)} consecutive frames with {model_name}")
        
        try:
        
            # Get model from cache
            model_files = {
                'YOLOv8m': 'yolov8m.pt',
                'YOLOv8x': 'yolov8x.pt', 
                'YOLOv9e': 'yolov9e.pt',
                'YOLOv10x': 'yolov10x.pt',
                'YOLOv11l': 'yolo11l.pt',
                'YOLOv11x': 'yolo11x.pt',
                'YOLOv12l': 'yolo12l.pt',
                'YOLOv12x': 'yolo12x.pt',
            }
            
            model_file = model_files.get(model_name, f"{model_name}.pt")
            key = f"ultra::{model_file}::{params['half']}"
            
            if key not in self.model_cache:
                try:
                    self.model_cache[key] = YOLO(model_file)
                except Exception:
                    self.model_cache[key] = YOLO("yolov8m.pt")  # fallback
                    
            model = self.model_cache[key]
            
            # Batch inference - MUCH faster than individual calls
            start_time = time.time()
            with torch.inference_mode():
                results = model.predict(
                    source=frame_list,  # List of frames for batch processing
                    imgsz=params['imgsz'], 
                    conf=params['conf'], 
                    iou=params['primary_iou'],
                    half=params['half'], 
                    verbose=False,
                    classes=[0],  # person only
                    max_det=params['max_det'],
                    augment=params['use_tta']
                )
            
            batch_inference_time = time.time() - start_time
            avg_inference_time = batch_inference_time / len(frame_list)
            
            print(f"🚀 BATCH: {len(frame_list)} frames in {batch_inference_time*1000:.1f}ms (avg: {avg_inference_time*1000:.1f}ms/frame)")
            
            # Process batch results
            batch_results = []
            for i, (result, frame) in enumerate(zip(results, frame_list)):
                h, w = frame.shape[:2]
                img_area = float(max(1, h * w))
                min_h = max(1.0, h * params['min_height_ratio'])
                min_area = max(1.0, img_area * params['min_area_ratio'])
                
                # Get polygon ROI if enabled
                poly = (cfg or {}).get("polygon_cropping", {}) or {}
                roi_polygon = None
                if bool(poly.get("enabled", False)):
                    pts = poly.get("points") or []
                    if len(pts) >= 3:
                        pixel_points = self._polygon_points_to_pixels(pts, w, h)
                        roi_polygon = [(pt[0], pt[1]) for pt in pixel_points]
                
                def _inside_roi_center(bbox, polygon):
                    if not polygon:
                        return True
                    cx = int((bbox[0] + bbox[2]) * 0.5)
                    cy = int((bbox[1] + bbox[3]) * 0.5)
                    poly_np = np.array(polygon, dtype=np.int32).reshape(-1, 1, 2)
                    return cv2.pointPolygonTest(poly_np, (cx, cy), False) >= 0
                
                detections = []
                for b in getattr(result, 'boxes', []) or []:
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
                    if all(_iou(cand, k) < params['secondary_iou'] for k in kept):
                        kept.append(cand)
                
                batch_results.append({
                    'people_count': len(kept),
                    'detections': kept,
                    'inference_time': avg_inference_time,
                    'model_name': model_name,
                    'frame_index': i
                })
                
                print(f"   Frame {i+1}: {len(kept)} people detected")
            
            return batch_results
            
        except Exception as e:
            print(f"❌ Batch inference error: {e}")
            # Fallback to individual inference
            batch_results = []
            for i, frame in enumerate(frame_list):
                result = self._infer(frame, cfg)
                if isinstance(result, dict):
                    batch_results.append({
                        'people_count': result.get('people_count', 0),
                        'detections': result.get('detections', []),
                        'inference_time': result.get('inference_time', 0.0),
                        'model_name': result.get('model_name', 'Fallback'),
                        'frame_index': i
                    })
                else:
                    batch_results.append({
                        'people_count': result if isinstance(result, int) else 0,
                        'detections': [],
                        'inference_time': 0.0,
                        'model_name': 'Legacy',
                        'frame_index': i
                    })
            return batch_results

    def get_consecutive_frames(self, n=5):
        """Get N most recent consecutive frames from buffer"""
        with self.frame_buffer_lock:
            if len(self.consecutive_frame_buffer) >= n:
                # Get last N frames
                recent_frames = list(self.consecutive_frame_buffer)[-n:]
                return [frame_data['frame'] for frame_data in recent_frames], recent_frames
            else:
                print(f"⚠️  Only {len(self.consecutive_frame_buffer)} frames available, need {n}")
                return [], []

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
        Optimized wait time estimation - only recalculate when people count changes.
        Uses cached wait time when people count is same as last time.
        """
        # If people count hasn't changed, return cached wait time
        if count == self._last_wait_people_count:
            print(f"🕒 Using cached wait time: {self._cached_wait_time}")
            return self._cached_wait_time
        
        # People count changed, calculate new wait time
        rt = (cfg or {}).get("runtime", {}) or {}
        tmin = float(rt.get("per_person_time_min", 22))
        tmax = float(rt.get("per_person_time_max", 26))
        
        if count > 0:
            if tmax < tmin:
                tmax = tmin
            per_person_seconds = random.randint(int(tmin), int(tmax))
            new_wait_time = count * per_person_seconds
        else:
            new_wait_time = 0.0
        
        # Cache the new values
        self._last_wait_people_count = count
        self._cached_wait_time = new_wait_time
        
        return new_wait_time

    # ---------------- debug image saving ----------------
    
    def _ensure_debug_folders(self, detection_session_name):
        """Create debug folders for session-specific structure"""
        debug_dir = os.path.join(os.path.dirname(__file__), "..", "debug_images")
        
        # Create session-specific folder structure
        session_dir = os.path.join(debug_dir, detection_session_name)
        input_dir = os.path.join(session_dir, "input")
        output_dir = os.path.join(session_dir, "output")
        
        print(f"🐛 DEBUG: Creating folders at {debug_dir}")
        print(f"🐛 DEBUG: Input dir: {input_dir}")
        print(f"🐛 DEBUG: Output dir: {output_dir}")
        
        os.makedirs(input_dir, exist_ok=True)
        os.makedirs(output_dir, exist_ok=True)
        
        print(f"🐛 DEBUG: Folders created successfully")
        return input_dir, output_dir
    
    def _draw_bounding_boxes(self, image, detections, model_name=""):
        """Draw bounding boxes on image for debug visualization"""
        if not detections:
            return image
            
        debug_image = image.copy()
        
        for i, detection in enumerate(detections):
            bbox = detection.get('bbox', [])
            confidence = detection.get('confidence', 0.0)
            
            if len(bbox) >= 4:
                x1, y1, x2, y2 = map(int, bbox)
                
                # Draw bounding box (green color)
                cv2.rectangle(debug_image, (x1, y1), (x2, y2), (0, 255, 0), 2)
                
                # Draw confidence score
                label = f"Person {i+1}: {confidence:.2f}"
                label_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)[0]
                
                # Background rectangle for text
                cv2.rectangle(debug_image, (x1, y1 - label_size[1] - 10), 
                            (x1 + label_size[0], y1), (0, 255, 0), -1)
                
                # Text
                cv2.putText(debug_image, label, (x1, y1 - 5), 
                          cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)
        
        # Add summary text
        summary = f"Model: {model_name} | Count: {len(detections)}"
        cv2.putText(debug_image, summary, (10, 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
        
        return debug_image
    
    def _save_debug_session(self, frames_data, final_people_count, model_name=""):
        """Save all frames from a detection session in organized folders"""
        print(f"🐛 DEBUG SESSION: enabled={self.debug_enabled}, final_count={final_people_count}, frames={len(frames_data)}")
        
        if not self.debug_enabled:
            print(f"🐛 DEBUG: Not enabled, skipping session")
            return
            
        if final_people_count < 1:
            print(f"🐛 DEBUG: No people in final count ({final_people_count}), skipping session")
            return
            
        # Check probability once for the entire session
        rand_val = random.randint(0, 100)
        print(f"🐛 DEBUG: Session probability check: {rand_val} vs {self.debug_probability}")
        if rand_val >= self.debug_probability:
            print(f"🐛 DEBUG: Session probability check failed ({rand_val} >= {self.debug_probability}), skipping")
            return
            
        try:
            # Create session folder name
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")[:-3]
            session_name = f"detection_{self.debug_images_saved+1:03d}_count_{final_people_count}_{timestamp}"
            
            input_dir, output_dir = self._ensure_debug_folders(session_name)
            
            # Save all frames in the session
            for i, frame_data in enumerate(frames_data):
                frame = frame_data['frame']
                detections = frame_data['detections']
                people_count = frame_data['people_count']
                
                # Save input image
                input_filename = f"frame_{i+1}.jpg"
                input_path = os.path.join(input_dir, input_filename)
                cv2.imwrite(input_path, frame)
                
                # Save output image with bounding boxes
                output_image = self._draw_bounding_boxes(frame, detections, f"{model_name}_frame_{i+1}")
                output_filename = f"frame_{i+1}.jpg"
                output_path = os.path.join(output_dir, output_filename)
                cv2.imwrite(output_path, output_image)
                
                print(f"   Saved frame {i+1}: {people_count} people")
            
            self.debug_images_saved += 1
            print(f"🐛 DEBUG: Saved debug session #{self.debug_images_saved} with {len(frames_data)} frames")
            print(f"   Session folder: {session_name}")
            
        except Exception as e:
            print(f"❌ Error saving debug session: {e}")

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
        """Optimized frame capture thread (based on legacy QMS optimizations)"""
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
                cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)  # Minimize buffer lag
                try:
                    cap.set(cv2.CAP_PROP_TIMEOUT, 10000)
                except:
                    pass
                    
            self._frame_capture = cap
            self.connected = cap is not None and cap.isOpened()
            
            # Optimized frame capture loop - use FPS from config
            cfg = self.get_config()
            target_fps = int((cfg or {}).get("runtime", {}).get("frame_read_fps", 30))
            frame_interval = 1.0 / max(10, int(target_fps))
            
            while self.capture_running and cap and cap.isOpened():
                try:
                    ret, frame = cap.read()
                    if ret and frame is not None:
                        cfg = self.get_config()
                        
                        # Create display frame with overlay (for preview)
                        display_frame = self._apply_display_overlay(frame, cfg)
                        
                        # Create detection frame with cropping (for AI processing)
                        detection_frame = self._apply_crop_for_detection(frame, cfg)
                        
                        # LEGACY QMS OPTIMIZATION: Smart queue management
                        # Always discard old frames, keep only the latest (non-blocking)
                        try:
                            # Remove old frame if queue not empty (legacy QMS pattern)
                            if not self.frame_queue.empty():
                                try:
                                    self.frame_queue.get_nowait()  # Discard old frame
                                except queue.Empty:
                                    pass
                            # Put new frame (non-blocking, timeout very short)
                            self.frame_queue.put(display_frame, timeout=0.01)
                            
                            # Store both frames for different uses (minimal locking)
                            with self.lock:
                                self.last_frame = display_frame.copy()  # For preview
                                self.last_detection_frame = detection_frame.copy()  # For AI
                                self.last_frame_ts = time.time()
                                
                            # Add frame to consecutive buffer for batch inference
                            with self.frame_buffer_lock:
                                self.consecutive_frame_buffer.append({
                                    'frame': detection_frame.copy(),
                                    'timestamp': time.time(),
                                    'frame_id': len(self.consecutive_frame_buffer)
                                })
                                
                        except queue.Full:
                            # Queue full - skip this frame (legacy QMS behavior)
                            pass
                            
                    else:
                        # Reconnect on read failure (legacy QMS pattern)
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
                    
                # LEGACY QMS OPTIMIZATION: Precise FPS timing
                time.sleep(max(0.0, frame_interval))
                
        finally:
            if cap is not None:
                try:
                    cap.release()
                except:
                    pass
            self.connected = False
            print(f"🎬 Optimized frame capture thread ended")
    
    def get_latest_frame(self):
        """Get latest frame from queue (non-blocking like legacy QMS)"""
        try:
            return self.frame_queue.get_nowait()
        except queue.Empty:
            return None
    
    def get_latest_detection_frame(self):
        """Get latest detection frame (non-blocking, optimized for AI processing)"""
        with self.lock:
            if self.last_detection_frame is not None:
                return self.last_detection_frame.copy()
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
        """Optimized detection loop (based on legacy QMS optimizations)"""
        last_det = 0.0
        frame_count = 0
        
        while not self.stop_flag and self.running:
            cfg = self.get_config()
            
            # Check if connection config changed - restart frame capture if needed
            new_stream_url = self._build_stream_url(cfg)
            if new_stream_url != self.current_stream_url:
                print(f"🔄 DETECTION: Connection config changed, restarting frame capture...")
                self._start_frame_capture(cfg)

            # LEGACY QMS OPTIMIZATION: Non-blocking frame access
            # Always get the LATEST frame, never wait for old frames
            detection_frame = self.get_latest_detection_frame()
                    
            if detection_frame is None:
                # No frame available yet - short sleep and continue (non-blocking)
                time.sleep(0.05)  # Shorter sleep for better responsiveness
                continue
                
            frame_count += 1
            now = time.time()

            # LEGACY QMS OPTIMIZATION: Detection interval timing
            # Clamp detection interval 0.1..10.0
            det_interval = float((cfg or {}).get("runtime", {}).get("detection_interval", 1.0))
            det_interval = max(0.1, min(10.0, det_interval))
            
            time_since_last = now - last_det

            # LEGACY QMS OPTIMIZATION: Process at intervals but on FRESH frames only
            if time_since_last >= det_interval:
                print(f"🤖 Starting detection inference (interval: {det_interval}s)")
                
                # Track total cycle start time
                cycle_start_time = time.time()
                
                # Check if using Median stabilization - run detection on 3 consecutive frames
                stab_method = (cfg or {}).get("count_stabilization", {}).get("method", "EMA")
                
                if stab_method == "Median":
                    # Get configurable frame count (2-5, default 3)
                    median_frame_count = int((cfg or {}).get("count_stabilization", {}).get("median_frame_count", 3))
                    median_frame_count = max(2, min(5, median_frame_count))  # Clamp to 2-5 range
                    
                    print(f"🔄 Using BATCH Median stabilization - processing {median_frame_count} consecutive frames...")
                    
                    # Get truly consecutive frames from buffer
                    frame_list, frame_metadata = self.get_consecutive_frames(median_frame_count)
                    
                    if frame_list:
                        # Batch inference on consecutive frames - MUCH faster!
                        batch_results = self._infer_batch(frame_list, cfg)
                        
                        # Extract people counts and calculate average
                        frame_results = [result['people_count'] for result in batch_results]
                        total_inference_time = sum(result['inference_time'] for result in batch_results)
                        
                        # Calculate average people count
                        raw_pc = int(round(sum(frame_results) / len(frame_results))) if frame_results else 0
                        inference_time = total_inference_time / len(frame_results) if frame_results else 0.0
                        model_name = f'{median_frame_count}-frame-batch'
                        
                        print(f"🔍 BATCH {median_frame_count}-Frame results: {frame_results} → Average: {raw_pc}")
                        
                        # Prepare frames for debug saving (with actual detections from batch)
                        frames_for_debug = []
                        for i, (result, frame_data) in enumerate(zip(batch_results, frame_metadata)):
                            frames_for_debug.append({
                                'frame': frame_data['frame'].copy(),
                                'detections': result['detections'],
                                'people_count': result['people_count'],
                                'inference_time': result['inference_time']
                            })
                        
                        # Save debug session with all frames
                        if frames_for_debug:
                            self._save_debug_session(frames_for_debug, raw_pc, model_name)
                        
                        # Use empty detections list for median mode (we don't track individual detections)
                        detections = []
                        
                    else:
                        print("⚠️  Not enough consecutive frames in buffer, falling back to single frame")
                        # Fallback to single frame detection
                        detection_frame = self.get_latest_detection_frame()
                        if detection_frame is not None:
                            result = self._infer(detection_frame, cfg)
                            if isinstance(result, dict):
                                raw_pc = result.get('people_count', 0)
                                detections = result.get('detections', [])
                                inference_time = result.get('inference_time', 0.0)
                                model_name = 'single-frame-fallback'
                            else:
                                raw_pc = result if isinstance(result, int) else 0
                                detections = []
                                inference_time = 0.0
                                model_name = 'legacy-fallback'
                        else:
                            raw_pc = 0
                            detections = []
                            inference_time = 0.0
                            model_name = 'no-frame'
                    
                else:
                    # Single frame detection for EMA and Rolling methods
                    detection_results = self._infer(detection_frame, cfg)
                    print(f"🔍 Detection results: {detection_results if isinstance(detection_results, dict) else f'count={detection_results}'}")
                    
                    if isinstance(detection_results, dict):
                        raw_pc = detection_results.get('people_count', 0)
                        detections = detection_results.get('detections', [])
                        inference_time = detection_results.get('inference_time', 0.0)
                        model_name = detection_results.get('model_name', 'Unknown')
                        
                        # Debug images are now saved via batch system in median mode
                    else:
                        raw_pc = detection_results
                        detections = []
                        inference_time = 0.0
                        model_name = 'Legacy'
                        
                        # Debug images are now saved via batch system in median mode
                
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

                # Calculate total cycle time
                total_cycle_time = time.time() - cycle_start_time

                with self.lock:
                    self._raw_people_count = raw_pc
                    self.people_count = final_count
                    self.wait_time = wait_seconds
                    self.total_cycle_time = total_cycle_time
                    
                    # Get stabilization method for debug output
                    stab_method = (cfg or {}).get("count_stabilization", {}).get("method", "EMA")
                    
                    print(f"📊 DETECTION UPDATE:")
                    print(f"   Raw count: {raw_pc}")
                    print(f"   Stabilized ({stab_method}): {st_pc}")
                    print(f"   Final count: {final_count}")
                    print(f"   Wait time: {wait_seconds:.1f}s")
                    print(f"   Per-frame inference: {inference_time*1000:.1f}ms")
                    print(f"   Total cycle time: {total_cycle_time*1000:.1f}ms")
                    
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


STATE = SharedState()