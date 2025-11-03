# ---------- trackers/centroid_obstruction.py ----------
from typing import List, Dict, Tuple, Optional
import numpy as np
import math
from collections import defaultdict

def _centroid_xyxy(box: np.ndarray) -> Tuple[float, float]:
    x1, y1, x2, y2 = box
    return ( (x1 + x2) * 0.5, (y1 + y2) * 0.5 )

def _iou(a: np.ndarray, b: np.ndarray) -> float:
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    inter_x1, inter_y1 = max(ax1, bx1), max(ay1, by1)
    inter_x2, inter_y2 = min(ax2, bx2), min(ay2, by2)
    iw, ih = max(0.0, inter_x2 - inter_x1), max(0.0, inter_y2 - inter_y1)
    inter = iw * ih
    if inter <= 0.0:
        return 0.0
    area_a = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
    area_b = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
    union = area_a + area_b - inter
    return inter / union if union > 0 else 0.0

def _project_world(cx: float, cy: float, H: Optional[np.ndarray], scale_mpp: float) -> Tuple[float, float]:
    """
    If H is provided, project image (pixels) -> world plane (meters).
    Else, fall back to uniform pixel->meter scaling (approx).
    """
    if H is not None:
        p = np.array([cx, cy, 1.0], dtype=float)
        w = H @ p
        if abs(w[2]) < 1e-9:
            return (float('nan'), float('nan'))
        return (w[0] / w[2], w[1] / w[2])
    else:
        # Simple fallback: treat pixels as meters * scale (rough)
        return (cx * scale_mpp, cy * scale_mpp)

class CentroidObstructionTracker:
    """
    Lightweight centroid-based tracker + velocity + obstruction detection.

    - Tracks are matched by IoU first (greedy), then by centroid distance.
    - Velocity is computed from homography-projected centroids if H is given,
      otherwise using a simple meter-per-pixel scale.
    - Mean velocity is computed from active tracks with valid velocity.
    - Obstruction is declared if v_i < alpha * mean_v for at least hold_time seconds.
    """
    def __init__(
        self,
        alpha: float = 0.3,           # obstruction if v < alpha * mean_v
        hold_time: float = 3.0,       # seconds below threshold to confirm obstruction
        fps_hint: float = 30.0,       # used when dt<=0 edge-cases
        iou_thresh: float = 0.3,      # for greedy IoU matching
        dist_thresh: float = 80.0,    # pixels; for centroid distance fallback
        max_miss_time: float = 1.5,   # seconds before track is dropped if unseen
        H: Optional[np.ndarray] = None,
        scale_mpp: float = 0.05       # meters-per-pixel if H is None
    ):
        self.alpha = float(alpha)
        self.hold_time = float(hold_time)
        self.fps_hint = float(fps_hint)
        self.iou_thresh = float(iou_thresh)
        self.dist_thresh = float(dist_thresh)
        self.max_miss_time = float(max_miss_time)
        self.H = H
        self.scale_mpp = float(scale_mpp)

        self._next_id = 1
        self._tracks: Dict[int, dict] = {}  # id -> {box, cx, cy, wx, wy, last_t, miss_t, v, hold, status}
        self._prev_time: Optional[float] = None

        # expose latest summaries
        self.latest_velocities: Dict[int, float] = {}
        self.latest_obstructions: Dict[int, bool] = {}

    def _match(self, dets: np.ndarray) -> Tuple[Dict[int, int], List[int], List[int]]:
        """
        Greedy 2-stage matching:
          1) IoU-based (≥ iou_thresh)
          2) centroid distance (≤ dist_thresh)
        Returns:
          matched: dict[track_id] = det_index
          unmatched_tracks: list[track_id]
          unmatched_dets: list[det_index]
        """
        track_ids = list(self._tracks.keys())
        matched: Dict[int, int] = {}
        if len(track_ids) == 0 or len(dets) == 0:
            return {}, track_ids, list(range(len(dets)))

        # Stage 1: IoU
        remaining_tracks = set(track_ids)
        remaining_dets = set(range(len(dets)))

        # Build all IoUs
        ious = []
        for tid in track_ids:
            tbox = self._tracks[tid]["box"]
            for j in range(len(dets)):
                iou = _iou(np.array(tbox, dtype=float), dets[j])
                if iou >= self.iou_thresh:
                    ious.append((iou, tid, j))
        # Greedy high->low IoU
        ious.sort(reverse=True, key=lambda x: x[0])
        for iou, tid, j in ious:
            if tid in remaining_tracks and j in remaining_dets:
                matched[tid] = j
                remaining_tracks.remove(tid)
                remaining_dets.remove(j)

        # Stage 2: centroid distance
        if remaining_tracks and remaining_dets:
            # Precompute centroids
            det_cens = [ _centroid_xyxy(dets[j]) for j in remaining_dets ]
            det_list = list(remaining_dets)
            for tid in list(remaining_tracks):
                tcx, tcy = self._tracks[tid]["cx"], self._tracks[tid]["cy"]
                # find nearest det by euclidean distance
                best_j = -1
                best_d = float("inf")
                for idx, j in enumerate(det_list):
                    dcx, dcy = det_cens[idx]
                    d = math.hypot(dcx - tcx, dcy - tcy)
                    if d < best_d:
                        best_d, best_j = d, j
                if best_j != -1 and best_d <= self.dist_thresh and best_j in remaining_dets:
                    matched[tid] = best_j
                    remaining_tracks.remove(tid)
                    remaining_dets.remove(best_j)

        return matched, list(remaining_tracks), list(remaining_dets)

    def update(self, detections_xyxy: List[List[float]], frame_time_s: float) -> Dict[int, dict]:
        """
        Update tracker with current-frame detections.

        Args:
          detections_xyxy: list of [x1,y1,x2,y2] in *full-frame* pixels.
          frame_time_s   : timestamp in seconds.

        Returns:
          tracks dict: id -> {"box","cx","cy","wx","wy","v","status"}
        """
        dets = np.array(detections_xyxy, dtype=float) if len(detections_xyxy) else np.zeros((0,4), dtype=float)

        # Compute dt
        if self._prev_time is None:
            self._prev_time = frame_time_s
            # initialize tracks from dets
            for box in dets:
                cx, cy = _centroid_xyxy(box)
                wx, wy = _project_world(cx, cy, self.H, self.scale_mpp)
                self._tracks[self._next_id] = dict(
                    box=box.tolist(), cx=cx, cy=cy, wx=wx, wy=wy,
                    last_t=frame_time_s, miss_t=0.0, v=float('nan'),
                    hold=0.0, status="normal"
                )
                self._next_id += 1
            self._refresh_summaries()
            return self._export_public_state()

        dt = frame_time_s - self._prev_time
        if dt <= 0:
            dt = 1.0 / self.fps_hint
        self._prev_time = frame_time_s

        # Match
        matched, un_tracks, un_dets = self._match(dets)

        # Update matched tracks (compute velocity)
        for tid, j in matched.items():
            box = dets[j]
            cx, cy = _centroid_xyxy(box)
            wx, wy = _project_world(cx, cy, self.H, self.scale_mpp)

            prev_wx, prev_wy = self._tracks[tid]["wx"], self._tracks[tid]["wy"]
            if (not math.isnan(prev_wx)) and (not math.isnan(prev_wy)) and (not math.isnan(wx)) and (not math.isnan(wy)):
                dx, dy = (wx - prev_wx), (wy - prev_wy)
                v = math.hypot(dx, dy) / dt  # m/s
            else:
                v = float('nan')

            # Save
            tr = self._tracks[tid]
            tr.update(dict(box=box.tolist(), cx=cx, cy=cy, wx=wx, wy=wy, last_t=frame_time_s))
            tr["miss_t"] = 0.0
            tr["v"] = v

        # Age unmatched tracks
        to_delete = []
        for tid in un_tracks:
            tr = self._tracks[tid]
            tr["miss_t"] += dt
            # keep last known box/coords; velocity not updated this frame
            if tr["miss_t"] > self.max_miss_time:
                to_delete.append(tid)
        for tid in to_delete:
            del self._tracks[tid]

        # Create new tracks for unmatched detections
        for j in un_dets:
            box = dets[j]
            cx, cy = _centroid_xyxy(box)
            wx, wy = _project_world(cx, cy, self.H, self.scale_mpp)
            self._tracks[self._next_id] = dict(
                box=box.tolist(), cx=cx, cy=cy, wx=wx, wy=wy,
                last_t=frame_time_s, miss_t=0.0, v=float('nan'),
                hold=0.0, status="normal"
            )
            self._next_id += 1

        # Compute mean velocity (only valid v)
        valid_vs = [ tr["v"] for tr in self._tracks.values() if tr["miss_t"] == 0.0 and not math.isnan(tr["v"]) ]
        mean_v = float(np.mean(valid_vs)) if len(valid_vs) > 0 else 0.0

        # Obstruction logic per track
        for tid, tr in self._tracks.items():
            v = tr["v"]
            if mean_v > 0.0 and (not math.isnan(v)) and (v < self.alpha * mean_v) and tr["miss_t"] == 0.0:
                tr["hold"] += dt
                if tr["hold"] >= self.hold_time:
                    tr["status"] = "obstruction"
            else:
                # reset if recovered or invalid
                tr["hold"] = 0.0
                if tr["status"] == "obstruction":
                    # keep status until it recovers for one full frame at/above threshold
                    if mean_v > 0.0 and (not math.isnan(v)) and (v >= self.alpha * mean_v):
                        tr["status"] = "normal"
                else:
                    tr["status"] = "normal"

        self._refresh_summaries()
        return self._export_public_state()

    def _refresh_summaries(self):
        self.latest_velocities = { tid: tr["v"] for tid, tr in self._tracks.items() if not math.isnan(tr["v"]) }
        self.latest_obstructions = { tid: (tr["status"] == "obstruction") for tid, tr in self._tracks.items() }

    def _export_public_state(self) -> Dict[int, dict]:
        # minimal view for external consumers
        return {
            tid: {
                "box": tr["box"],
                "v": tr["v"],
                "status": tr["status"],
                "cx": tr["cx"], "cy": tr["cy"],
                "wx": tr["wx"], "wy": tr["wy"],
            }
            for tid, tr in self._tracks.items()
        }

