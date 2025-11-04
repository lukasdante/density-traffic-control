# file: multi_rtsp_yolo_pure_gst.py
import site; site.addsitedir('/usr/lib/python3/dist-packages')
import os, time, threading, queue
from dataclasses import dataclass
from pathlib import Path
from typing import Optional
import numpy as np
import cv2

# --- GStreamer (PyGObject) ---
import gi # type: ignore
gi.require_version('Gst', '1.0')
from gi.repository import Gst # type: ignore
Gst.init(None)

from PySide6.QtGui import QImage, QPixmap
from PySide6.QtCore import Qt, Slot
import torch
from ultralytics import YOLO
from classes import Camera, ProxyCamera
from trackers import CentroidObstructionTracker
from publisher import publish

# --- CONFIG ---
MODEL_PATH = str(Path(__file__).parent.parent / "yolo" / "best.pt")  # adjust to your model
MODEL_PATH = "yolo11n.pt"
WINDOW = "YOLO 2x2 (low-latency)"
TILE_W, TILE_H = 640, 360
CONFIDENCE = 0.30
FONT = cv2.FONT_HERSHEY_SIMPLEX

# Draw settings
DRAW_THICKNESS = 2
DRAW_FONT_SCALE = 0.6

class GstCam:
    """Minimal wrapper around a GStreamer pipeline with an appsink."""
    def __init__(self, pipeline_str: str):
        self.pipeline = Gst.parse_launch(pipeline_str)
        self.appsink = self.pipeline.get_by_name("appsink")
        if self.appsink is None:
            raise RuntimeError("appsink not found in pipeline")
        self.appsink.set_property("emit-signals", False)
        self.appsink.set_property("sync", False)
        self.appsink.set_property("max-buffers", 1)
        self.appsink.set_property("drop", True)
        self.pipeline.set_state(Gst.State.PLAYING)

    def read(self):
        # Non-blocking: returns (ok, frame) or (False, None) if no fresh frame
        sample = self.appsink.emit("try-pull-sample", 0)
        if sample is None:
            return False, None
        buf = sample.get_buffer()
        caps = sample.get_caps()
        s = caps.get_structure(0)
        w = s.get_value('width'); h = s.get_value('height')
        ok, mapinfo = buf.map(Gst.MapFlags.READ)
        if not ok:
            return False, None
        try:
            # IMPORTANT: make a WRITEABLE copy so OpenCV can draw on it
            frame = np.frombuffer(mapinfo.data, dtype=np.uint8).reshape((h, w, 3)).copy()
        finally:
            buf.unmap(mapinfo)
        return True, frame

    def release(self):
        self.pipeline.set_state(Gst.State.NULL)

@dataclass
class Cam:
    url: str
    cap: GstCam
    q: "queue.Queue[np.ndarray]"
    last: Optional[np.ndarray] = None
    alive: bool = True

class Inferencer:
    def __init__(self, label_widgets, target_sources, stop_flag, latency_tracker, display_settings, experimental_settings):
        self.label_widgets = label_widgets
        self.target_sources = target_sources
        self.stop_flag = stop_flag
        self.latency_tracker = latency_tracker
        self.display_settings = display_settings
        self.experimental_settings = experimental_settings

        self.tile_w = self.experimental_settings.get("tile_w", TILE_W)
        self.tile_h = self.experimental_settings.get("tile_h", TILE_H)
        self.confidence = self.experimental_settings.get("confidence", CONFIDENCE)

        self.cycle_green = 15
        self.cycle_yellow = 3
        self.total_cycle = (self.cycle_green + self.cycle_yellow) * len(self.target_sources)
        self.start_time = time.time()

        self.last_green_lane = None
        self.lanes = ["A", "B", "C", "D"]

        self.tracker = CentroidObstructionTracker(
            alpha=0.30,          # v_i < 0.30 * mean_v  -> suspect
            hold_time=3.0,       # seconds below threshold to declare obstruction
            fps_hint=30.0,
            iou_thresh=0.3,
            dist_thresh=80.0,
            max_miss_time=1.5,
            H=None,              # <-- set your 3x3 homography here when ready
            scale_mpp=0.05       # fallback if H is None
        )

    def build_pipeline(self, rtsp_url: str, w: int | None = None, h: int | None = None) -> str:
        size_caps = ""
        if w and h:
            # only apply videoscale if you intentionally want to resize
            size_caps = f" ! videoscale ! video/x-raw,width={w},height={h},format=BGR"
        else:
            # keep native resolution, no scaling
            size_caps = " ! video/x-raw,format=BGR"
        
        return (
            f'rtspsrc location="{rtsp_url}" latency=50 protocols=tcp ! '
            f'rtph264depay ! h264parse ! avdec_h264 ! videoconvert{size_caps} ! '
            f'appsink name=appsink caps=video/x-raw,format=BGR drop=true max-buffers=1 sync=false'
        )



    def reader_thread(self,cam: Cam):
        while cam.alive:
            frame_start = time.perf_counter()
            ok, frame = cam.cap.read()
            if not ok or frame is None:
                time.sleep(0.005)
                continue
            if not cam.q.empty():
                try:
                    _ = cam.q.get_nowait()
                except queue.Empty:
                    pass
            capture_latency = time.perf_counter() - frame_start
            self.latency_tracker.record_capture(capture_latency)
            cam.q.put(frame)


    def make_grid(self, frames, tile_w, tile_h):
        tiles = []
        for f in frames:
            if f is None:
                tiles.append(np.zeros((tile_h, tile_w, 3), dtype=np.uint8))
            else:
                tiles.append(cv2.resize(f, (tile_w, tile_h)))
        while len(tiles) < 4:
            tiles.append(np.zeros((tile_h, tile_w, 3), dtype=np.uint8))
        top = np.hstack((tiles[0], tiles[1]))
        bot = np.hstack((tiles[2], tiles[3]))
        return np.vstack((top, bot))

    def draw_boxes(self, frame, boxes, class_names):
        """
        boxes: (N,6) array [x1,y1,x2,y2,conf,cls]
        Draws boxes per class depending on self.display_settings['bounding_boxes'] flags.
        """
        if boxes is None or len(boxes) == 0:
            return frame

        bbox_settings = self.display_settings.get("bounding_boxes", {})
        CLASS_COLORS = {
            "obstructions": (0, 165, 255),   # Orange
            "two-wheeled":  (0, 255, 255),   # Yellow
            "light":        (0, 255, 0),     # Green
            "heavy":        (255, 0, 255),   # Purple
        }
            # --- Get current frame dimensions (O(1) operation) ---
        h, w = frame.shape[:2]

        # --- Compute scale factor relative to 1280x720 baseline ---
        base_w, base_h = 1280, 720
        scale = ((w / base_w) + (h / base_h)) / 2.0

        # --- Adaptive visual parameters ---
        font_scale = 0.6 * scale
        box_thickness = max(1, int(0.8 * scale))
        text_thickness = max(1, int(1.2 * scale))
        padding = int(3 * scale)

        for x1, y1, x2, y2, conf, cls in boxes:
            cls = int(cls)
            name = class_names[cls] if cls < len(class_names) else f"cls_{cls}"
            
            if not bbox_settings.get(name, True):
                continue

            color = CLASS_COLORS.get(name, (255, 255, 255))  # default white

            x1, y1, x2, y2 = map(int, (x1, y1, x2, y2))
            label = f"{name} {float(conf):.2f}"
            
            

            # Draw main bounding box
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, box_thickness)
            
            if self.display_settings["osd"].get("annotations", False):
                (tw, th), _ = cv2.getTextSize(label, FONT, font_scale, text_thickness)
                cv2.rectangle(frame, (x1, max(0, y1 - th - 6)), (x1 + tw + 6, y1), color, -1)
                cv2.putText(frame, label, (x1 + padding, max(0, y1 - padding)), FONT, font_scale, (0, 0, 0), text_thickness, cv2.LINE_AA)

        return frame


    def _init_yolo(self):
        """Initialize YOLO model and send to device."""
        device = 0 if torch.cuda.is_available() else "cpu"
        model = YOLO(MODEL_PATH)
        model.to(device=device)

        if device != "cpu":
            model.fuse()
            try:
                model.model.half()
            except Exception:
                pass

        return model, device

    def _prepare_roi_crops(self, frames):
        """
        Returns a list of masked images (ROI only) and metadata for coordinate restoration.
        Inference will happen *only inside* the polygon region.
        """
        cropped_imgs, roi_info = [], []

        for i, frame in enumerate(frames):
            cam = self.target_sources[i]
            if frame is None:
                cropped_imgs.append(None)
                roi_info.append((0, 0, 0, 0))
                continue

            h, w = frame.shape[:2]

            # --- Default: full frame if no ROI ---
            if not hasattr(cam, "roi") or cam.roi is None:
                cropped_imgs.append(frame)
                roi_info.append((0, 0, w, h))
                continue

            # --- Convert normalized ROI points to pixel coordinates ---
            roi_pts = np.array([(int(x * w), int(y * h)) for x, y in cam.roi], dtype=np.int32)

            # --- Create mask (white = inside ROI, black = outside) ---
            mask = np.zeros((h, w), dtype=np.uint8)
            cv2.fillPoly(mask, [roi_pts], 255)

            # --- Apply mask: zero out everything outside ROI ---
            masked_frame = cv2.bitwise_and(frame, frame, mask=mask)

            # --- Crop bounding rect for faster inference ---
            x1, y1 = np.min(roi_pts, axis=0)
            x2, y2 = np.max(roi_pts, axis=0)

            # Clamp to bounds
            x1, y1 = max(0, x1), max(0, y1)
            x2, y2 = min(w, x2), min(h, y2)

            roi_crop = masked_frame[y1:y2, x1:x2].copy()
            cropped_imgs.append(roi_crop)
            roi_info.append((x1, y1, x2 - x1, y2 - y1))

        return cropped_imgs, roi_info


    def _batch_infer(self, model, batch_imgs, device, idx_map):
        """Run YOLO strictly within each camera's ROI only."""
        roi_info_list = []
        self._roi_info_by_frame = {}
        self._result_index_by_frame = {}

        # Crop first (per valid frame)
        cropped_imgs, roi_info_list = self._prepare_roi_crops(batch_imgs)
        self._roi_info_by_frame = {frame_idx: info for frame_idx, info in zip(idx_map, roi_info_list)}
        self._result_index_by_frame = {frame_idx: local_idx for local_idx, frame_idx in enumerate(idx_map)}

        # Remove None frames (keeps batch clean)
        valid_imgs = [img for img in cropped_imgs if img is not None]
        if not valid_imgs:
            self._latest_velocities = {}
            self._latest_obstructions = {}
            return []

        inf_start = time.perf_counter()
        preds = model.predict(
            valid_imgs,
            imgsz=max(self.tile_w, self.tile_h),
            device=device,
            half=(device != "cpu"),
            conf=self.confidence,
            iou=0.45,
            verbose=False
        )
        self.latency_tracker.record_inference(time.perf_counter() - inf_start)

        # ---- Build full-frame boxes for tracking ----
        frame_time = time.time()
        all_boxes_xyxy = []  # flatten across the valid frames (your display code already remaps by index)

        for local_idx, frame_idx in enumerate(idx_map):
            if local_idx >= len(preds):
                break
            r = preds[local_idx]
            if r.boxes is not None and len(r.boxes) > 0:
                xyxy = r.boxes.xyxy.cpu().numpy()
                # shift back to full-frame coords using stored ROI offsets
                x_off, y_off, _, _ = self._roi_info_by_frame.get(frame_idx, (0, 0, 0, 0))
                if xyxy.size > 0:
                    xyxy[:, [0, 2]] += x_off
                    xyxy[:, [1, 3]] += y_off
                    for box in xyxy:
                        all_boxes_xyxy.append(box.tolist())

        # ---- Update tracker with all full-frame detections this cycle ----
        tracks_state = self.tracker.update(all_boxes_xyxy, frame_time)
        # Optionally keep latest summaries for drawing elsewhere
        self._latest_velocities = self.tracker.latest_velocities
        self._latest_obstructions = self.tracker.latest_obstructions

        # Print only those declared as obstructions
        for tid, is_obstructed in self.tracker.latest_obstructions.items():
            if is_obstructed:
                v = self.tracker.latest_velocities.get(tid, float("nan"))
                print(f"[OBSTRUCTION] Track {tid} velocity={v:.2f} m/s")

        return preds

    def _display_frames(self, frames, results, idx_map):
        """Draw bounding boxes and OSD text per frame after ROI-only inference."""
        display_start = time.perf_counter()

        roi_info_map = getattr(self, "_roi_info_by_frame", {})
        result_idx_map = getattr(self, "_result_index_by_frame", {})

        if results:
            class_names = results[0].names

            for i, frame in enumerate(frames):
                if frame is None:
                    continue

                num_objs = 0

                result_idx = result_idx_map.get(i)
                if result_idx is not None and result_idx < len(results):
                    r = results[result_idx]
                    x_off, y_off, _, _ = roi_info_map.get(i, (0, 0, 0, 0))
                    if r.boxes is not None and len(r.boxes) > 0:
                        xyxy = r.boxes.xyxy.cpu().numpy()
                        conf = r.boxes.conf.unsqueeze(1).cpu().numpy()
                        cls = r.boxes.cls.unsqueeze(1).cpu().numpy()
                        boxes = np.concatenate([xyxy, conf, cls], axis=1)

                        # Shift detections back into full-frame coordinates
                        boxes[:, [0, 2]] += x_off
                        boxes[:, [1, 3]] += y_off

                        # Draw boxes
                        frame = self.draw_boxes(frame, boxes, class_names)
                        frame = self.draw_obstruction(frame)
                        num_objs = len(r.boxes.cls)

                # --- Draw OSD ---
                cam = self.target_sources[i]
                frame = self.draw_osd(frame, cam, num_objs)

                # --- Update frame list (for display) ---
                frames[i] = frame

        # --- Display all frames in their QLabel widgets ---
        for i, frame in enumerate(frames):
            if frame is None:
                continue

            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            h, w, ch = rgb.shape
            qimg = QImage(rgb.data, w, h, ch * w, QImage.Format_RGB888)
            pix = QPixmap.fromImage(qimg)
            self.label_widgets[i].setPixmap(pix.scaled(
                self.label_widgets[i].size(),
                Qt.KeepAspectRatio,
                Qt.SmoothTransformation
            ))

        # --- Record latency ---
        self.latency_tracker.record_display(time.perf_counter() - display_start)


    def draw_osd(self, frame, cam, num_objs):
        """
        Draw On-Screen Display (OSD) info (camera name, location, congestion)
        scaled relative to video resolution/aspect ratio.
        """
        if not hasattr(self, "display_settings"):
            return frame

        display_settings = getattr(self, "display_settings", {})
        osd = display_settings.get("osd", {})

        h, w, _ = frame.shape

        # --- Compute scale relative to reference resolution ---
        # Reference = 1280x720; change to your usual baseline
        base_w, base_h = self.tile_w, self.tile_h
        scale_factor = ((w / base_w) + (h / base_h)) / 2.0  # geometric mean works too

        # --- Adjustable visual parameters scaled ---
        font = cv2.FONT_HERSHEY_SIMPLEX
        base_scale = 0.45
        scale = base_scale * scale_factor
        thickness = max(1, int(1 * scale_factor))
        line_h = int(25 * scale_factor)
        padding = int(10 * scale_factor)
        margin_x = int(10 * scale_factor)
        margin_y = int(10 * scale_factor)
        color = (255, 255, 255)
        bg_color = (30, 30, 30)
        alpha = 0.3

        # --- Generate text lines ---
        lines = []
        if osd.get("name", False):
            lines.append(f"Cam: {getattr(cam, 'name', 'Unknown')}")
        if osd.get("location", False):
            lines.append(f"Loc: {getattr(cam, 'location', 'Unknown')}")
        if osd.get("estimated_congestion", False):
            try:
                max_cap = int(getattr(cam, 'max_cap', 100))
            except (ValueError, TypeError):
                max_cap = 100
            congestion_ratio = num_objs / max_cap if max_cap > 0 else 0
            lines.append(f"Congestion: {congestion_ratio:.2f}")
        
        if hasattr(cam, "roi") and cam.roi is not None and osd.get("roi", False):
            h, w, _ = frame.shape
            pts = np.array([(int(x*w), int(y*h)) for x, y in cam.roi], dtype=np.int32)
            cv2.polylines(frame, [pts], isClosed=True, color=(255, 0, 0), thickness=2)

        if not lines:
            return frame

        # --- Dynamic rectangle size ---
        text_widths = [cv2.getTextSize(t, font, scale, thickness)[0][0] for t in lines]
        rect_width = padding * 2 + max(text_widths)
        rect_height = padding * 2 + line_h * len(lines)

        # --- Translucent background ---
        overlay = frame.copy()
        cv2.rectangle(
            overlay,
            (margin_x, margin_y),
            (margin_x + rect_width, margin_y + rect_height),
            bg_color,
            -1
        )
        frame = cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0)

        # --- Draw text lines ---
        y = margin_y + padding + int(line_h * 0.8)
        for text in lines:
            cv2.putText(
                frame, text,
                (margin_x + padding, y),
                font, scale, color, thickness, cv2.LINE_AA
            )
            y += line_h
        

        # --- Traffic Signal (top-right corner) ---
        radius = int(15 * scale_factor)
        gap = int(8 * scale_factor)
        x_right = w - margin_x - radius
        y_top = margin_y + radius + 5

        signal = self._get_signal_state(cam)
        colors = {"red": (0, 0, 255), "yellow": (0, 255, 255), "green": (0, 255, 0)}

        for i, c in enumerate(["red", "yellow", "green"]):
            cy = y_top + i * (2 * radius + gap)
            circle_color = colors[c] if signal == c else (80, 80, 80)
            cv2.circle(frame, (x_right, cy), radius, circle_color, -1)
            cv2.circle(frame, (x_right, cy), radius, (0, 0, 0), 2)


        return frame


    @Slot(dict)
    def on_display_settings_changed(self, display_dict):
        self.display_settings = display_dict.copy()
        print(f"[INFO] OSD settings updated: {self.display_settings}")
    
    @Slot(dict)
    def on_experimental_settings_changed(self, experimental_dict):
        self.experimental_settings = experimental_dict.copy()
        print(f"[INFO] Inference settings updated: {self.experimental_settings}")
    
    @Slot(list)
    def on_camera_data_changed(self, new_cameras):
        """Update camera source list without restarting inference."""
        self.target_sources = new_cameras
        print("[INFO] Inference camera sources updated.")


    def run(self):
        if not self.target_sources:
            return
        if isinstance(self.target_sources[0], Camera):
            self.run_rtsp_mode()
        if isinstance(self.target_sources[0], ProxyCamera):
            self.run_video_mode()

    def run_video_mode(self):
        model, device = self._init_yolo()
        caps = []

        # OpenCV-based sources (ProxyCamera.file_path)
        for proxy in self.target_sources:
            cap = cv2.VideoCapture(proxy.file_path)
            if not cap.isOpened():
                print(f"[ERROR] Cannot open video: {proxy.file_path}")
                continue
            caps.append(cap)

        try:
            while not self.stop_flag.is_set():
                capture_start = time.perf_counter()
                frames = []
                for cap in caps:
                    ret, frame = cap.read()
                    if not ret:
                        cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                        ret, frame = cap.read()
                    
                    if ret and frame is not None:
                        frame = cv2.resize(frame, (self.tile_w, self.tile_h), interpolation=cv2.INTER_AREA)

                    frames.append(frame)
                self.latency_tracker.record_capture(time.perf_counter() - capture_start)

                batch_imgs, idx_map = [], []
                for i, f in enumerate(frames):
                    if f is not None:
                        batch_imgs.append(f)
                        idx_map.append(i)

                results = []
                if batch_imgs:
                    results = self._batch_infer(model, batch_imgs, device, idx_map)

                self._display_frames(frames, results, idx_map)

        finally:
            for cap in caps:
                cap.release()

    def run_rtsp_mode(self):
        """ Runs the inferencer in RTSP mode. """
        model, device = self._init_yolo()
        cams = []

        for cam in self.target_sources:
            cap = GstCam(self.build_pipeline(cam.full_link, self.tile_w, self.tile_h))
            cam = Cam(url=cam.full_link, cap=cap, q=queue.Queue(maxsize=1))
            t = threading.Thread(target=self.reader_thread, args=(cam,), daemon=True)
            t.start()
            cams.append(cam)

        try:
            while not self.stop_flag.is_set():
                capture_start = time.perf_counter()
                frames = []
                for cam in cams:
                    try:
                        cam.last = cam.q.get(timeout=0.02)
                    except queue.Empty:
                        pass
                    frames.append(cam.last)
                self.latency_tracker.record_capture(time.perf_counter() - capture_start)

                batch_imgs, idx_map = [], []
                for i, f in enumerate(frames):
                    if f is not None:
                        batch_imgs.append(f)
                        idx_map.append(i)

                results = []

                # Batch infer
                if batch_imgs:
                    results = self._batch_infer(model, batch_imgs, device, idx_map)
                
                self._display_frames(frames, results, idx_map)

        finally:
            for cam in cams:
                cam.alive = False
                try:
                    cam.cap.release()
                except Exception:
                    pass
    def draw_obstruction(self, frame):
        """
        Draw bounding boxes for current obstructions detected by the tracker.

        Parameters
        ----------
        frame : np.ndarray
            The video frame to draw on (BGR).

        Returns
        -------
        np.ndarray
            The frame with obstruction boxes drawn.
        """
        if not hasattr(self, "_latest_obstructions"):
            return frame
        if not self._latest_obstructions:
            return frame

        # Scale-based sizing
        h, w = frame.shape[:2]
        base_w, base_h = 1280, 720
        scale = ((w / base_w) + (h / base_h)) / 2.0
        font_scale = 0.6 * scale
        thickness = max(2, int(1.2 * scale))
        text_thickness = max(1, int(1.2 * scale))
        padding = int(3 * scale)

        for tid, is_obstruction in self._latest_obstructions.items():
            if not is_obstruction:
                continue
            tr = self.tracker._tracks.get(tid)
            if not tr:
                continue

            x1, y1, x2, y2 = map(int, tr["box"])
            color = (0, 165, 255)  # Orange
            label = "OBSTRUCTION"

            # Draw box
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, thickness)

            # Draw label background
            (tw, th), _ = cv2.getTextSize(label, FONT, font_scale, text_thickness)
            cv2.rectangle(frame, (x1, max(0, y1 - th - 6)), (x1 + tw + 6, y1), color, -1)
            cv2.putText(frame, label, (x1 + padding, max(0, y1 - padding)),
                        FONT, font_scale, (0, 0, 0), text_thickness, cv2.LINE_AA)

        return frame

    def _get_signal_state(self, cam):
        """
        Determine the light color (green/yellow/red) based on the rotating lane cycle.
        Only triggers publish() when a *new* lane turns green.
        """
        elapsed = (time.time() - self.start_time) % self.total_cycle
        lane_index = int(elapsed // (self.cycle_green + self.cycle_yellow))
        lane = self.lanes[lane_index]  # active lane this cycle
        phase_time = elapsed % (self.cycle_green + self.cycle_yellow)

        # Determine color for the *active* lane
        if phase_time < self.cycle_green:
            current_state = "green"
        else:
            current_state = "yellow"

        # Determine each camera’s current state
        cam_index = self.lanes.index(cam.name[-1]) if cam.name[-1] in self.lanes else 0
        cam_lane = self.lanes[cam_index]
        state = "red"

        if cam_lane == lane:
            state = current_state

        # --- Publish event only when the green lane changes ---
        if current_state == "green" and self.last_green_lane != lane:
            self.last_green_lane = lane
            print(f"[INFO] Lane {lane} turned GREEN → publishing to RPi...")
            try:
                publish(lane)
            except Exception as e:
                print(f"[ERROR] MQTT publish failed for lane {lane}: {e}")

        return state
