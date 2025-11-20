# app/state.py
import os
import json
import time
import threading
import random
import pickle
from collections import deque
from typing import Any, Dict, Optional, List
from datetime import datetime
import cv2
import numpy as np
from ultralytics import YOLO
import torch
import yaml

CONFIG_PATH = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "data", "config.json")
)

# Removed QueuePredictor - main app uses simple random calculation


class SharedState:
    def __init__(self):
        self.lock = threading.RLock()
        self.config = self._load_config()  # type: Dict[str, Any]

        # Live stats
        self.people_count = 0               # stabilized + adjusted
        self._raw_people_count = 0          # raw detector output

        self.wait_time = 0.0                # seconds
        self.last_frame = None              # BGR frame (display with overlay)
        self.last_detection_frame = None    # BGR frame (cropped for detection)
        self.last_frame_ts = 0.0
        
        # Wait time caching to avoid recalculation when people count is same
        self._last_wait_people_count = -1   # Track last people count for wait time
        self._cached_wait_time = 0.0        # Cached wait time for same people count
        
        # Frame drop fallback caching - use last successful detection when frames drop
        self._last_successful_people_count = 0    # Last successful people count
        self._last_successful_wait_time = 0.0     # Last successful wait time
        self._last_successful_timestamp = 0.0     # When last successful detection occurred
        
        # Debug image saving
        self.debug_enabled = False          # Enable/disable debug image saving
        self.debug_probability = 10         # Probability (0-100) to save debug images
        self.debug_images_saved = 0         # Counter for saved debug images
        
        print(f"🐛 DEBUG INIT: enabled={self.debug_enabled}, probability={self.debug_probability}%")
        
        # Enhanced features
        self.detection_history = deque(maxlen=200)  # Detection history
        self.total_detections = 0
        self.average_inference_time = 0.0
        self.total_cycle_time = 0.0         # Total detection cycle time
        
        # Tracking mode max_delta state
        self._tracker_last_count = 0        # Last committed count for tracking mode
        
        # Tracking median averaging buffer
        self._tracking_median_buffer = deque(maxlen=10)  # Buffer for median averaging in tracking mode
        self._tracking_ids_buffer = deque(maxlen=10)     # Buffer for tracking IDs in median mode
        
        # Display update gating
        self.last_display_update_time = 0.0
        self._internal_count = 0            # Latest stable count (decoupled from display)

        # Runtime states
        self.connected = False
        self.stop_flag = False
        self.running = False
        self.thread = None                  # type: Optional[threading.Thread]
        
        # Frame capture system (optimized for tracking - no queue needed)
        # Tracking mode always uses latest frame via self.last_detection_frame
        self.capture_thread = None          # type: Optional[threading.Thread]
        self.capture_running = False
        self.current_stream_url = None
        
        # Consecutive frame buffer for batch inference
        self.consecutive_frame_buffer = deque(maxlen=10)  # Store last 10 consecutive frames
        self.frame_buffer_lock = threading.Lock()  # Thread-safe access to frame buffer
        
        # FPS monitoring and warning system
        self.actual_fps = 0.0               # Measured actual camera FPS
        self.configured_fps = 0             # Target FPS from config
        self.fps_warning_active = False     # True if FPS below threshold (80%)
        self.fps_measurements = deque(maxlen=30)  # Rolling window for FPS calculation
        self.last_frame_time = None         # Last frame timestamp for FPS calculation

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
        Apply polygon masking for AI detection.
        Masks areas outside the polygon ROI.
        """
        poly = (cfg or {}).get("polygon_cropping", {}) or {}

        # Polygon mask - keeps same frame size, masks outside regions
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
        Apply visual overlay for preview display.
        Shows polygon ROI with dimmed areas outside the region.
        """
        poly = (cfg or {}).get("polygon_cropping", {}) or {}
        
        display_frame = frame.copy()
        
        # Polygon overlay with visual feedback
        if bool(poly.get("enabled", False)):
            pts = poly.get("points") or []
            if isinstance(pts, list) and len(pts) >= 3:
                h, w = frame.shape[:2]
                # Use IDENTICAL coordinate conversion as frontend polygon editor
                pixel_points = self._polygon_points_to_pixels(pts, w, h)
                if len(pixel_points) >= 3:
                    poly_pts = np.array(pixel_points, dtype=np.int32)
                    
                    # Create overlay with red polygon outline
                    overlay = display_frame.copy()
                    cv2.polylines(overlay, [poly_pts], isClosed=True, color=(0, 0, 255), thickness=2)
                    
                    # Create mask for polygon area (identical to detection mask)
                    mask = np.zeros((h, w), dtype=np.uint8)
                    cv2.fillPoly(mask, [poly_pts], 255)
                    
                    # Dim everything to 30%
                    dim = (display_frame * 0.3).astype(display_frame.dtype)
                    
                    # Show overlay (with red outline) inside polygon, dimmed outside
                    display_frame = np.where(mask[..., None] == 255, overlay, dim)
        
        return display_frame

    # ---------------- models ----------------

    def _get_model_params(self, cfg):
        """Extract common model parameters to avoid duplication"""
        runtime = (cfg or {}).get("runtime", {}) or {}
        return {
            'conf': 0.1,  # Get ALL detections from model, filter later
            'user_conf_threshold': float(runtime.get("confidence_threshold", 0.5)),
            'imgsz': int(runtime.get("imgsz", 1280)),
            'half': bool(runtime.get("half_precision", False)),
            'max_det': int(runtime.get("max_det", 100)),
            'primary_iou': float(runtime.get("primary_iou", 0.68)),
            'min_height_ratio': float(runtime.get("min_height_ratio", 0.018)),
            'min_area_ratio': float(runtime.get("min_area_ratio", 0.00015))
        }



    # ---------------- Tracking methods ----------------
    
    def _write_tracker_yaml(self, tracker_type, track_cfg):
        """
        Write tracker YAML file with current configuration.
        """

        
        # Common parameters for both trackers
        yaml_config = {
            'tracker_type': tracker_type,
            'track_high_thresh': float(track_cfg.get('track_high_thresh', 0.6)),
            'track_low_thresh': float(track_cfg.get('track_low_thresh', 0.1)),
            'new_track_thresh': float(track_cfg.get('new_track_thresh', 0.6)),
            'track_buffer': int(track_cfg.get('track_buffer', 30)),
            'match_thresh': float(track_cfg.get('match_thresh', 0.8)),
            'fuse_score': bool(track_cfg.get('fuse_score', True))
        }
        
        # Add BoT-SORT specific parameters
        if tracker_type == 'botsort':
            yaml_config.update({
                'gmc_method': track_cfg.get('gmc_method', 'sparseOptFlow'),
                'proximity_thresh': float(track_cfg.get('proximity_thresh', 0.5)),
                'appearance_thresh': float(track_cfg.get('appearance_thresh', 0.8)),
                'with_reid': bool(track_cfg.get('with_reid', False)),
                'model': 'auto'
            })
        
        # Write YAML file
        yaml_path = f"{tracker_type}.yaml"
        try:
            with open(yaml_path, 'w') as f:
                yaml.dump(yaml_config, f, default_flow_style=False, sort_keys=False)
        except Exception as e:
            print(f"Warning: Could not write {yaml_path}: {e}")
    
    def _track_frame(self, frame, cfg):
        """
        Track objects in single frame using ByteTrack or BoT-SORT.
        Returns tracking results with persistent IDs across frames.
        """
        if frame is None or frame.size == 0:
            return None
            
        runtime = (cfg or {}).get("runtime", {}) or {}
        model_name = str(runtime.get("selected_model", "YOLOv11x"))
        params = self._get_model_params(cfg)
        
        # Get model from cache (same as inference)
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
        
        try:
            # Get tracking configuration
            stab_cfg = (cfg or {}).get("count_stabilization", {}) or {}
            tracker_type = stab_cfg.get("tracker_type", "bytetrack")
            track_cfg = stab_cfg.get(tracker_type, {}) or {}
            
            # Write YAML file with current config (tracker handles confidence filtering internally)
            self._write_tracker_yaml(tracker_type, track_cfg)
            
            start_time = time.time()
            tracker_yaml = f"{tracker_type}.yaml"
            with torch.inference_mode():
                results = model.track(
                    source=frame,
                    persist=True,              
                    tracker=tracker_yaml,
                    imgsz=params['imgsz'], 
                    conf=params['conf'], 
                    iou=params['primary_iou'],
                    half=params['half'], 
                    verbose=False,
                    classes=[0],  # person only
                    max_det=params['max_det'],
                    augment=False
                )

            
            inference_time = time.time() - start_time
            
            if results and len(results) > 0:
                result = results[0]
                
                # Frame is already masked/cropped before tracking
                # Tracker handles confidence filtering via YAML
                # Just extract the tracked objects
                valid_tracks = []
                if result.boxes is not None and result.boxes.id is not None:
                    boxes = result.boxes.xyxy.cpu().numpy()
                    confidences = result.boxes.conf.cpu().numpy()
                    track_ids = result.boxes.id.cpu().numpy().astype(int)
                    
                    for box, conf, track_id in zip(boxes, confidences, track_ids):
                        x1, y1, x2, y2 = box
                        valid_tracks.append({
                            'track_id': int(track_id),
                            'confidence': float(conf),
                            'bbox': [int(x1), int(y1), int(x2), int(y2)]
                        })
                
                return {
                    'valid_tracks': valid_tracks,
                    'inference_time': inference_time,
                    'model_name': model_name
                }
            
            return {
                'valid_tracks': [],
                'inference_time': inference_time,
                'model_name': model_name
            }
            
        except Exception as e:
            print(f"❌ Tracking error: {e}")
            return None
    # ---------------- wait-time ----------------

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
    
    def _save_empty_queue_frame(self, frame, timestamp):
        """
        Save frame when queue is empty (0 people) in tracker mode with 0.25% probability.
        This helps collect edge case data without overwhelming storage.
        """
        try:
            # Create directory for empty queue frames
            empty_queue_dir = os.path.join(os.path.dirname(__file__), "..", "empty_queue_frames")
            os.makedirs(empty_queue_dir, exist_ok=True)
            
            # Create filename with timestamp
            dt = datetime.fromtimestamp(timestamp)
            filename = f"empty_queue_{dt.strftime('%Y%m%d_%H%M%S_%f')[:-3]}.jpg"
            filepath = os.path.join(empty_queue_dir, filename)
            
            # Save frame
            cv2.imwrite(filepath, frame)
            
            # Count saved frames
            saved_count = len([f for f in os.listdir(empty_queue_dir) if f.endswith('.jpg')])
            print(f"📸 TRACKER: Saved empty queue frame #{saved_count} (0 people, 0.25% probability)")
            print(f"   File: {filename}")
            
        except Exception as e:
            print(f"❌ Error saving empty queue frame: {e}")

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
            
            # Initialize FPS tracking
            self.configured_fps = target_fps
            self.last_frame_time = None
            frame_count = 0
            
            while self.capture_running and cap and cap.isOpened():
                try:
                    ret, frame = cap.read()
                    if ret and frame is not None:
                        # FPS measurement and warning system
                        current_time = time.time()
                        if self.last_frame_time is not None:
                            frame_delta = current_time - self.last_frame_time
                            if frame_delta > 0:
                                instantaneous_fps = 1.0 / frame_delta
                                self.fps_measurements.append(instantaneous_fps)
                                
                                # Calculate rolling average FPS every 30 frames
                                if len(self.fps_measurements) >= 30:
                                    self.actual_fps = sum(self.fps_measurements) / len(self.fps_measurements)
                                    
                                    # Check if actual FPS is below 80% of configured FPS
                                    fps_threshold = self.configured_fps * 0.8
                                    if self.actual_fps < fps_threshold:
                                        if not self.fps_warning_active:
                                            print(f"⚠️  FPS WARNING: Camera delivering {self.actual_fps:.1f} FPS, configured for {self.configured_fps} FPS")
                                            print(f"⚠️  Actual FPS is {(self.actual_fps/self.configured_fps*100):.1f}% of configured - consider reducing frame_read_fps")
                                            self.fps_warning_active = True
                                    else:
                                        if self.fps_warning_active:
                                            print(f"✅ FPS OK: Camera now delivering {self.actual_fps:.1f} FPS (target: {self.configured_fps} FPS)")
                                            self.fps_warning_active = False
                        
                        self.last_frame_time = current_time
                        frame_count += 1
                        
                        cfg = self.get_config()
                        
                        # Create display frame with overlay (for preview)
                        display_frame = self._apply_display_overlay(frame, cfg)
                        
                        # Create detection frame with cropping (for AI processing)
                        detection_frame = self._apply_crop_for_detection(frame, cfg)
                        
                        # TRACKING OPTIMIZATION: Always store latest frame (no queue needed)
                        # Tracking mode processes frames as fast as they arrive
                        with self.lock:
                            self.last_frame = display_frame.copy()  # For legacy compatibility
                            self.last_detection_frame = detection_frame.copy()  # For tracking/detection
                            self.last_frame_ts = time.time()
                            
                        # Add frame to consecutive buffer for batch inference (if needed for other modes)
                        with self.frame_buffer_lock:
                            self.consecutive_frame_buffer.append({
                                'frame': detection_frame.copy(),
                                'timestamp': time.time(),
                                'frame_id': len(self.consecutive_frame_buffer)
                            })
                            
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
            
            # Reset FPS tracking
            self.actual_fps = 0.0
            self.fps_warning_active = False
            self.fps_measurements.clear()
            self.last_frame_time = None
            
            print(f"🎬 Optimized frame capture thread ended")
    
    def get_latest_detection_frame(self):
        """Get latest detection frame (non-blocking, optimized for AI processing)"""
        with self.lock:
            if self.last_detection_frame is not None:
                return self.last_detection_frame.copy()
        return None

    # ---------------- worker loop ----------------

    def _reset_tracking_state(self):
        """Clear tracker - ByteTrack/BoT-SORT handles its own state with persist=True"""
        # Tracker state is managed by ultralytics, nothing to reset
        # Reset max_delta tracking state
        self._tracker_last_count = 0
        print("🔄 Tracker will reset on next detection start")
    
    def start(self):
        if self.thread and self.thread.is_alive():
            self.running = True
            print("🚀 Detection already running, resuming...")
            return
        print("🚀 Starting detection system...")
        
        # Reset tracking state for fresh start
        self._reset_tracking_state()
        
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
        
        # Reset tracking state
        self._reset_tracking_state()
        
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
            
            # Get current time (needed for both frame drop and normal processing)
            now = time.time()
                    
            if detection_frame is None:
                # Frame drop detected - use cached values and skip to next detection interval
                det_interval = float((cfg or {}).get("runtime", {}).get("detection_interval", 1.0))
                print(f"❌ DEBUG: Frame drop detected, using cached values and skipping to next detection interval ({det_interval}s)")
                
                with self.lock:
                    # Use cached values from last successful detection
                    cached_count = getattr(self, '_last_successful_people_count', 0)
                    cached_wait = getattr(self, '_last_successful_wait_time', 0.0)
                    cached_timestamp = getattr(self, '_last_successful_timestamp', 0.0)
                    
                    self.people_count = cached_count
                    self.wait_time = cached_wait
                    
                    age_seconds = now - cached_timestamp if cached_timestamp > 0 else 0
                    print(f"🔄 DEBUG: Using cached values: people_count={cached_count}, wait_time={cached_wait:.1f}s (cached {age_seconds:.1f}s ago)")
                    
                # Skip to next detection interval
                last_det = now
                continue
                
            frame_count += 1

            # LEGACY QMS OPTIMIZATION: Detection interval timing
            # Clamp detection interval 0.1..10.0
            det_interval = float((cfg or {}).get("runtime", {}).get("detection_interval", 1.0))
            det_interval = max(0.1, min(10.0, det_interval))
            
            time_since_last = now - last_det
            
            # ============================================================
            # TRACKING MODE: Continuous tracking, update display at intervals
            # ============================================================
            # Track EVERY frame (tracker handles persistence)
            tracking_result = self._track_frame(detection_frame, cfg)
            
            if tracking_result:
                # Update display at detection_interval
                time_since_last = now - last_det
                if time_since_last >= det_interval:
                    # Trust ByteTrack/BoT-SORT - count the tracks it returns
                    raw_track_count = len(tracking_result['valid_tracks'])
                    
                    # Apply max_delta protection for tracking mode (safety against sudden spikes)
                    cs = (cfg or {}).get("count_stabilization", {}) or {}
                    max_delta = int(cs.get("max_delta_per_detection", 1))
                    
                    delta = raw_track_count - self._tracker_last_count
                    if delta > max_delta:
                        people_count = self._tracker_last_count + max_delta
                        print(f"🛡️  TRACKING: Rate limited +{delta} to +{max_delta} (from {self._tracker_last_count} to {people_count}, raw: {raw_track_count})")
                    elif delta < -max_delta:
                        people_count = self._tracker_last_count - max_delta
                        print(f"🛡️  TRACKING: Rate limited {delta} to -{max_delta} (from {self._tracker_last_count} to {people_count}, raw: {raw_track_count})")
                    else:
                        people_count = raw_track_count
                    
                    # Update committed count for next detection
                    self._tracker_last_count = people_count
                    
                    # Extract track IDs early for buffer usage
                    track_ids = [t['track_id'] for t in tracking_result['valid_tracks']]
                    
                    # Apply batch-based median averaging if enabled
                    # Collects X frames, calculates average, displays, then starts fresh
                    median_enabled = bool(cs.get("median_enabled", False))
                    if median_enabled:
                        median_frame_size = int(cs.get("median_frame_size", 5))
                        median_frame_size = max(2, min(10, median_frame_size))  # Clamp to 2-10
                        
                        # Add current count and IDs to buffer
                        self._tracking_median_buffer.append(people_count)
                        self._tracking_ids_buffer.append(sorted(track_ids))
                        
                        # Check if buffer is full - calculate average and clear for fresh batch
                        if len(self._tracking_median_buffer) >= median_frame_size:
                            averaged_count = int(round(sum(self._tracking_median_buffer) / len(self._tracking_median_buffer)))
                            print(f"📊 BATCH MEDIAN: Collected {list(self._tracking_median_buffer)} → Average: {averaged_count} → Starting fresh batch")
                            people_count = averaged_count
                            self._internal_count = people_count  # Update internal truth
                            # Clear buffer for fresh batch
                            self._tracking_median_buffer.clear()
                            self._tracking_ids_buffer.clear()
                        else:
                            # Still collecting frames, use last known internal count
                            print(f"📊 COLLECTING: {len(self._tracking_median_buffer)}/{median_frame_size} frames collected, using last count: {self._internal_count}")
                            people_count = self._internal_count  # Use internal truth, not displayed value
                    else:
                        # Clear buffer if median is disabled
                        self._tracking_median_buffer.clear()
                        self._tracking_ids_buffer.clear()
                        self._internal_count = people_count  # Update internal truth
                    
                    # Save frame with 0.25% probability when queue is empty (tracker mode only)
                    if people_count == 0:
                        # 0.25% = 0.0025 probability
                        # Using random integer check: 1 out of 400 (0.25% = 1/400)
                        if random.randint(1, 400) == 1:
                            self._save_empty_queue_frame(detection_frame, now)
                    
                    # Calculate wait time
                    wait_seconds = self._estimate_wait(people_count, cfg)
                    
                    # Check display refresh interval
                    display_interval = float((cfg or {}).get("display_config", {}).get("refresh_interval", 0.0))
                    time_since_display_update = now - self.last_display_update_time
                    
                    should_update_display = True
                    if display_interval > 0 and time_since_display_update < display_interval:
                        should_update_display = False
                        print(f"⏳ DISPLAY: Holding update (elapsed {time_since_display_update:.1f}s < {display_interval}s)")
                    
                    if should_update_display:
                        with self.lock:
                            self.people_count = people_count
                            self.wait_time = wait_seconds
                            self.last_frame_ts = now
                            self.last_display_update_time = now
                            
                            # Cache successful values
                            self._last_successful_people_count = people_count
                            self._last_successful_wait_time = wait_seconds
                            self._last_successful_timestamp = now
                    else:
                        # Even if we don't update display, we update internal tracking state?
                        # Actually, self.people_count IS the display state.
                        # So we just don't update self.people_count.
                        pass
                    
                    # Check for duplicate IDs in current frame (sanity check)
                    seen = set()
                    duplicates = [x for x in track_ids if x in seen or seen.add(x)]
                    
                    print(f"📊 TRACKING UPDATE:")
                    print(f"   Active tracks: {sorted(track_ids)}")
                    if median_enabled:
                        print(f"   Buffer tracks: {list(self._tracking_ids_buffer)}")
                    if duplicates:
                        print(f"   ⚠️ DUPLICATE IDs: {duplicates}")
                    else:
                        print(f"   Duplicate IDs: None")
                        
                    print(f"   Raw track count: {raw_track_count}")
                    print(f"   Rate-limited count: {people_count} (max_delta: ±{max_delta})")
                    print(f"   Wait time: {wait_seconds:.1f}s ({wait_seconds/60:.1f} min)")
                    print(f"   Inference: {tracking_result['inference_time']*1000:.1f}ms")
                    
                    # Update telemetry
                    self.total_detections += 1
                    if len(self.detection_history) > 0:
                        recent_times = [r['inference_time'] for r in list(self.detection_history)[-10:]]
                        self.average_inference_time = sum(recent_times) / len(recent_times)
                    else:
                        self.average_inference_time = tracking_result['inference_time']
                    
                    # Add to detection history
                    tracker_name = (cfg or {}).get("count_stabilization", {}).get("tracker_type", "bytetrack").upper()
                    detection_record = {
                        'timestamp': now,
                        'people_count': people_count,
                        'wait_time': wait_seconds / 60.0,
                        'model_name': f"{tracker_name}-{tracking_result['model_name']}",
                        'confidence': float((cfg or {}).get("runtime", {}).get("confidence_threshold", 0.5)),
                        'inference_time': tracking_result['inference_time']
                    }
                    self.detection_history.append(detection_record)
                    
                    last_det = now
            
            # No sleep - process next frame immediately for continuous tracking
            continue


STATE = SharedState()