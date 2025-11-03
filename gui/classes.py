from dataclasses import dataclass
from collections import deque
import numpy as np
from typing import Literal

@dataclass
class BaseCamera:
    name: str
    location: str
    max_cap: int
    

@dataclass
class Camera(BaseCamera):
    ip_address: str
    username: str
    password: str
    rtsp_port: int
    channel: int
    roi: tuple[int, int, int] | None = None
    
    def __post_init__(self):
        self.full_link = f'rtsp://{self.username}:{self.password}@{self.ip_address}:{self.rtsp_port}/Streaming/Channels/{self.channel}'

    def __str__(self):
        return f"Camera: {self.name} at {self.full_link}"

@dataclass
class ProxyCamera(BaseCamera):
    file_path: str
    roi: tuple[int, int, int] | None = None


class LatencyTracker:
    def __init__(self, window=300):
        self.capture = deque(maxlen=window)
        self.inference = deque(maxlen=window)
        self.display = deque(maxlen=window)

    def record_capture(self, t): self.capture.append(t)
    def record_inference(self, t): self.inference.append(t)
    def record_display(self, t): self.display.append(t)

    def summary(self):
        def avg(q): return np.mean(q) if q else 0
        return {
            "capture": avg(self.capture),
            "inference": avg(self.inference),
            "display": avg(self.display),
            "total": avg(self.capture) + avg(self.inference) + avg(self.display)
        }
