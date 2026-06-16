# file: multi_rtsp_yolo_pure_gst.py
import os, time, threading, queue
from dataclasses import dataclass
from pathlib import Path
from typing import Optional
import numpy as np
import cv2

# --- GStreamer (PyGObject) ---
# Imported lazily so the app still launches (and the video-file demo works)
# on machines without system python3-gi / GStreamer. Only the live RTSP path
# requires it; see GstCam below.
Gst = None  # populated by _ensure_gst()


def _ensure_gst():
    """Import and initialise GStreamer on first use of the RTSP pipeline."""
    global Gst
    if Gst is not None:
        return Gst
    import site
    site.addsitedir('/usr/lib/python3/dist-packages')
    import gi  # type: ignore
    gi.require_version('Gst', '1.0')
    from gi.repository import Gst as _Gst  # type: ignore
    _Gst.init(None)
    Gst = _Gst
    return Gst

from PySide6.QtGui import QImage, QPixmap
from PySide6.QtCore import Qt, Slot
import torch
from ultralytics import YOLO
from classes import Camera, ProxyCamera
from publisher import publish

# --- CONFIG ---
MODEL_PATH = str(Path(__file__).parent.parent / "model" / "runs" / "yolo12-vehicles-overfit" / "weights" / "best.pt")
WINDOW = "YOLO 2x2 (low-latency)"
TILE_W, TILE_H = 640, 360
CONFIDENCE = 0.30
FONT = cv2.FONT_HERSHEY_SIMPLEX
MIN_GREEN = 15
MAX_GREEN = 85

# --- Emergency-vehicle preemption ---
# An ambulance/firetruck only triggers a green override when we are *confident*
# it is really an emergency vehicle: its detection confidence must clear
# EMERGENCY_CONFIDENCE AND it must persist for EMERGENCY_FRAMES consecutive
# inference cycles. This rejects single-frame false positives.
EMERGENCY_CONFIDENCE = 0.55
EMERGENCY_FRAMES = 5

# Draw settings
DRAW_THICKNESS = 2
DRAW_FONT_SCALE = 0.6

class GstCam:
    """Minimal wrapper around a GStreamer pipeline with an appsink."""
    def __init__(self, pipeline_str: str):
        _ensure_gst()
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
        self.min_green_time = MIN_GREEN
        self.max_green_time = MAX_GREEN
        self.emergency_confidence = float(self.experimental_settings.get("emergency_confidence", EMERGENCY_CONFIDENCE))
        self.emergency_frames = int(self.experimental_settings.get("emergency_frames", EMERGENCY_FRAMES))
        self.current_green_lane: Optional[str] = None
        self.last_switch_time = time.time()
        self.last_published_red_lane: Optional[str] = None
        self._last_vehicle_counts = {"horizontal": 0, "vertical": 0}
        self._class_ids: dict[str, set[int]] = {}
        self._class_names = {}
        # frame_idx -> (N,6) ndarray [x1,y1,x2,y2,conf,cls] in full-frame coords
        self._detections_by_frame: dict[int, np.ndarray] = {}
        self._emergency_streak = {"horizontal": 0, "vertical": 0}
        self._last_emergency_counts = {"horizontal": 0, "vertical": 0}
        self._has_vehicle_measurement = False
        self.yellow_duration = 3.0
        self._yellow_until = {"horizontal": 0.0, "vertical": 0.0}
        self._pending_lane: Optional[str] = None
        self._pending_start_time = 0.0

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
            "firetruck": (0, 0, 255),
            "ambulance": (0, 255, 255),
            "vehicle": (0, 255, 0),
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
            
            if not bbox_settings.get(name, False):
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
        """Initialize YOLO model and send to device.

        On an RTX 2060 the cheap, reliable wins are: FP16 weights, layer fusion,
        cuDNN autotuning (picks the fastest conv kernels for our fixed tile size)
        and TF32 matmuls. We also run one warm-up inference so the first real
        frame doesn't eat the kernel-compilation / allocator stall.
        """
        device = 0 if torch.cuda.is_available() else "cpu"
        model = YOLO(MODEL_PATH)
        model.to(device=device)

        if device != "cpu":
            # Tile size is constant frame-to-frame, so let cuDNN cache the best kernels.
            torch.backends.cudnn.benchmark = True
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True
            model.fuse()
            try:
                model.model.half()
            except Exception:
                pass

            # Warm up so the first measured frame reflects steady-state latency.
            try:
                imgsz = max(self.tile_w, self.tile_h)
                warmup = np.zeros((self.tile_h, self.tile_w, 3), dtype=np.uint8)
                model.predict(warmup, imgsz=imgsz, device=device, half=True,
                              conf=self.confidence, verbose=False)
            except Exception as exc:
                print(f"[WARN] YOLO warm-up skipped: {exc}")

        return model, device

    def _prepare_roi_crops(self, frames, idx_map):
        """
        Returns a list of masked images (ROI only) and metadata for coordinate restoration.
        Inference will happen *only inside* the polygon region.

        ``idx_map[i]`` is the real camera index for batch position ``i`` so the
        correct ROI is applied even when some cameras drop a frame.
        """
        cropped_imgs, roi_info = [], []

        for i, frame in enumerate(frames):
            cam = self.target_sources[idx_map[i]]
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
        """Run YOLO strictly within each camera's ROI and cache detections.

        Results are pulled off the GPU *once* per frame as a single (N,6) array
        [x1,y1,x2,y2,conf,cls] already shifted into full-frame coordinates. Both
        the signal logic and the renderer reuse this cache, so we no longer pay
        for two or three separate ``.cpu().numpy()`` transfers per frame.
        """
        self._detections_by_frame = {}

        # Crop to each camera's ROI first (batch_imgs are all non-None here).
        cropped_imgs, roi_info_list = self._prepare_roi_crops(batch_imgs, idx_map)
        if not cropped_imgs:
            return

        inf_start = time.perf_counter()
        preds = model.predict(
            cropped_imgs,
            imgsz=max(self.tile_w, self.tile_h),
            device=device,
            half=(device != "cpu"),
            conf=self.confidence,
            iou=0.45,
            verbose=False
        )
        self.latency_tracker.record_inference(time.perf_counter() - inf_start)

        if preds:
            self._class_names = preds[0].names

        for local_idx, r in enumerate(preds):
            frame_idx = idx_map[local_idx]
            boxes = getattr(r, "boxes", None)
            if boxes is None or len(boxes) == 0:
                continue
            # Single transfer: boxes.data is [x1,y1,x2,y2,conf,cls].
            det = boxes.data.detach().to("cpu").numpy().astype(np.float32, copy=True)
            x_off, y_off, _, _ = roi_info_list[local_idx]
            if det.size:
                det[:, [0, 2]] += x_off
                det[:, [1, 3]] += y_off
            self._detections_by_frame[frame_idx] = det

    def _display_frames(self, frames):
        """Draw bounding boxes and OSD text using the cached detections."""
        display_start = time.perf_counter()

        detections_map = self._detections_by_frame
        class_names = self._class_names or {}

        for i, frame in enumerate(frames):
            if frame is None:
                continue

            boxes = detections_map.get(i)
            num_objs = 0 if boxes is None else len(boxes)
            if boxes is not None and len(boxes):
                frame = self.draw_boxes(frame, boxes, class_names)

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

    def _resolve_class_ids(self, names) -> dict[str, set[int]]:
        """Map the labels we care about to their model class ids."""
        mapping: dict[str, set[int]] = {"vehicle": set(), "ambulance": set(), "firetruck": set()}
        if isinstance(names, dict):
            items = names.items()
        elif isinstance(names, (list, tuple)):
            items = enumerate(names)
        else:
            items = []

        for idx, label in items:
            try:
                key = int(idx)
            except (TypeError, ValueError):
                continue
            if label in mapping:
                mapping[label].add(key)
        return mapping

    def _direction_of(self, frame_idx: int) -> str:
        direction = getattr(self.target_sources[frame_idx], "direction", "vertical")
        return direction if direction in ("horizontal", "vertical") else "vertical"

    def _aggregate_detections(self):
        """Single pass over the cached detections → vehicle and emergency counts.

        Vehicles drive the density logic; an emergency count only includes
        ambulances/firetrucks whose confidence clears ``emergency_confidence``.
        """
        vehicle = {"horizontal": 0, "vertical": 0}
        emergency = {"horizontal": 0, "vertical": 0}

        if not self._class_ids and self._class_names:
            self._class_ids = self._resolve_class_ids(self._class_names)
        if not self._class_ids:
            return vehicle, emergency

        vehicle_ids = np.array(sorted(self._class_ids.get("vehicle", set())), dtype=int)
        emergency_ids = np.array(
            sorted(self._class_ids.get("ambulance", set()) | self._class_ids.get("firetruck", set())),
            dtype=int,
        )

        for frame_idx, det in self._detections_by_frame.items():
            if det is None or len(det) == 0:
                continue
            cls = det[:, 5].astype(int)
            conf = det[:, 4]
            direction = self._direction_of(frame_idx)
            if vehicle_ids.size:
                vehicle[direction] += int(np.count_nonzero(np.isin(cls, vehicle_ids)))
            if emergency_ids.size:
                emask = np.isin(cls, emergency_ids) & (conf >= self.emergency_confidence)
                emergency[direction] += int(np.count_nonzero(emask))

        return vehicle, emergency

    def _update_emergency_streak(self, emergency_counts):
        """Count consecutive cycles each lane has shown a confident emergency vehicle."""
        self._last_emergency_counts = emergency_counts
        for lane in ("horizontal", "vertical"):
            if emergency_counts.get(lane, 0) > 0:
                self._emergency_streak[lane] += 1
            else:
                self._emergency_streak[lane] = 0

    def _confirmed_emergency_lane(self) -> Optional[str]:
        """Lane whose emergency-vehicle presence has persisted long enough to trust."""
        confirmed = [lane for lane in ("vertical", "horizontal")
                     if self._emergency_streak.get(lane, 0) >= self.emergency_frames]
        if not confirmed:
            return None
        if len(confirmed) == 1:
            return confirmed[0]
        # Both lanes have emergencies: hold the current one, else pick the busier.
        if self.current_green_lane in confirmed:
            return self.current_green_lane
        return max(confirmed, key=lambda lane: self._last_emergency_counts.get(lane, 0))

    def _preferred_green_lane(self, vehicle_counts):
        horizontal = vehicle_counts.get("horizontal", 0)
        vertical = vehicle_counts.get("vertical", 0)

        if horizontal > vertical:
            return "horizontal"
        if vertical > horizontal:
            return "vertical"
        return None

    def _opposite_lane(self, lane: Optional[str]) -> Optional[str]:
        if lane == "horizontal":
            return "vertical"
        if lane == "vertical":
            return "horizontal"
        return None

    def _publish_red(self, lane: Optional[str], show_yellow: bool = True):
        if not lane:
            return
        if show_yellow:
            self._yellow_until[lane] = time.time() + self.yellow_duration
        else:
            self._yellow_until[lane] = 0.0
        try:
            publish(lane)
            self.last_published_red_lane = lane
        except Exception as exc:
            print(f"[ERROR] MQTT publish failed for lane {lane}: {exc}")

    def _schedule_lane_change(self, target_lane: Optional[str], now: float):
        if not target_lane or target_lane not in ("horizontal", "vertical"):
            return
        if self._pending_lane:
            return
        losing_lane = self._opposite_lane(target_lane)
        if losing_lane:
            self._publish_red(losing_lane, show_yellow=True)
        self.current_green_lane = None
        self._pending_lane = target_lane
        self._pending_start_time = now + self.yellow_duration

    def _update_signal_state(self, vehicle_counts, emergency_lane: Optional[str] = None):
        now = time.time()
        density_lane = self._preferred_green_lane(vehicle_counts)
        # An emergency lane outranks the busier lane when choosing the cold-start default.
        preferred_lane = emergency_lane or density_lane

        if self.current_green_lane is None and self._pending_lane is None and not self._has_vehicle_measurement:
            return

        if self._pending_lane:
            if now >= self._pending_start_time:
                self.current_green_lane = self._pending_lane
                self._pending_lane = None
                self.last_switch_time = now
            else:
                return

        if self.current_green_lane is None:
            default_lane = preferred_lane or "horizontal"
            self.current_green_lane = default_lane
            self.last_switch_time = now
            self._publish_red(self._opposite_lane(self.current_green_lane), show_yellow=False)
            return

        elapsed = now - self.last_switch_time
        target_lane = self.current_green_lane
        should_switch = False

        if emergency_lane and emergency_lane != self.current_green_lane:
            # Emergency preemption: switch immediately, bypassing the min-green hold.
            target_lane = emergency_lane
            should_switch = True
        elif emergency_lane == self.current_green_lane:
            # Hold green for the emergency lane; never time it out while it is active.
            should_switch = False
        elif elapsed >= self.max_green_time:
            forced_lane = self._opposite_lane(self.current_green_lane)
            if forced_lane:
                target_lane = forced_lane
                should_switch = True
        elif density_lane and density_lane != self.current_green_lane and elapsed >= self.min_green_time:
            target_lane = density_lane
            should_switch = True

        if should_switch and target_lane != self.current_green_lane:
            self._schedule_lane_change(target_lane, now)

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
            lines.append(f"Cam: {getattr(cam, 'name', 'Unknown')} [{getattr(cam, 'direction', 'X')[0].upper()}]")
        if osd.get("location", False):
            lines.append(f"Loc: {getattr(cam, 'location', 'Unknown')}")
        if osd.get("estimated_congestion", False):
            try:
                max_cap = int(getattr(cam, 'max_cap', 100))
            except (ValueError, TypeError):
                max_cap = 100
            congestion_ratio = num_objs / max_cap if max_cap > 0 else 0
            lines.append(f"Congestion: {congestion_ratio:.2f} ({num_objs} vehicles)")
        
        if hasattr(cam, "roi") and cam.roi is not None and osd.get("roi", False):
            h, w, _ = frame.shape
            pts = np.array([(int(x*w), int(y*h)) for x, y in cam.roi], dtype=np.int32)
            cv2.polylines(frame, [pts], isClosed=True, color=(255, 0, 0), thickness=2)

        draw_signal = osd.get("traffic_phase", False)

        if not lines and not draw_signal:
            return frame

        if lines:
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

        if draw_signal:
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
        self.tile_w = self.experimental_settings.get("tile_w", self.tile_w)
        self.tile_h = self.experimental_settings.get("tile_h", self.tile_h)
        self.confidence = self.experimental_settings.get("confidence", self.confidence)
        self.emergency_confidence = float(self.experimental_settings.get("emergency_confidence", self.emergency_confidence))
        self.emergency_frames = int(self.experimental_settings.get("emergency_frames", self.emergency_frames))
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

    def _process_frames(self, model, frames, device):
        """Infer, update the signal state and render one batch of frames.

        Shared by both the RTSP and the video-file pipelines so the two paths
        stay byte-for-byte identical in behaviour.
        """
        batch_imgs, idx_map = [], []
        for i, f in enumerate(frames):
            if f is not None:
                batch_imgs.append(f)
                idx_map.append(i)

        if batch_imgs:
            self._batch_infer(model, batch_imgs, device, idx_map)
        else:
            self._detections_by_frame = {}

        if idx_map:
            vehicle_counts, emergency_counts = self._aggregate_detections()
            self._last_vehicle_counts = vehicle_counts
            self._has_vehicle_measurement = True
            self._update_emergency_streak(emergency_counts)

        emergency_lane = self._confirmed_emergency_lane()
        self._update_signal_state(self._last_vehicle_counts, emergency_lane)
        self._display_frames(frames)

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

                self._process_frames(model, frames, device)

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

                self._process_frames(model, frames, device)

        finally:
            for cam in cams:
                cam.alive = False
                try:
                    cam.cap.release()
                except Exception:
                    pass

    def _get_signal_state(self, cam):
        """
        Determine the light color (green/yellow/red) for the camera lane based on the current
        adaptive signal state.
        """
        direction = getattr(cam, "direction", "vertical")
        direction = direction if direction in ("horizontal", "vertical") else "vertical"

        if time.time() < self._yellow_until.get(direction, 0.0):
            return "yellow"
        if self.current_green_lane is None:
            return "red"
        if direction == self.current_green_lane:
            return "green"
        return "red"
