# Density-Based Traffic Signal Control (YOLOv12 + ESP32)

Detects vehicles in up to four camera feeds, decides which of two traffic
lights (the **vertical** lane and the **horizontal** lane) should be green based
on traffic density, and lets emergency vehicles (ambulance / firetruck) preempt
the signal. Decisions are published over MQTT to ESP32-driven lights.

The trained detection model is **included** in this repository
(`model/runs/yolo12-vehicles-overfit/weights/best.pt`), so you do not need the
training data to run the app.

## Quick start

You need **Linux** (tested on Ubuntu 22.04), an **NVIDIA GPU**, and
[`uv`](https://docs.astral.sh/uv/). Install `uv` once:

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

Then, from this folder:

```bash
./run.sh
```

That runs `uv sync` (installs everything into a local `.venv`) and launches the
control panel. To do it manually instead:

```bash
uv sync
uv run python gui/main.py
```

### Live IP cameras (extra system packages)

The **video-file demo works out of the box**. To stream from live **RTSP IP
cameras**, also install GStreamer and the Python bindings (these are system
packages, not Python ones):

```bash
sudo apt install -y python3-gi python3-gi-cairo gir1.2-gstreamer-1.0 \
  gir1.2-gst-plugins-base-1.0 gstreamer1.0-plugins-good \
  gstreamer1.0-plugins-bad gstreamer1.0-plugins-ugly gstreamer1.0-libav \
  gstreamer1.0-tools
```

Then in the app: **Settings → Load settings → `configs/all_cameras.json`** (edit
the IPs/credentials for your cameras first).

## Documentation

Full guides are in [`docs/`](docs/) (LaTeX sources + compiled PDFs):

- `install.tex` — complete hardware + software setup (cameras, 2.4 GHz hotspot,
  ESP32 wiring and flashing).
- `usage.tex` — how to navigate and use the application.
- `optimization-report.tex` — the signal logic, emergency preemption, and
  performance work.

## Training data

The dataset and source videos are **not** in this repository (they are large).
They are hosted separately on Google Drive. Download them only if you want to
**re-train** the model; the app itself does not need them.

To re-train YOLOv12 on the demo data after downloading `tl-data/`:

```bash
./train_yolo12.sh
```

## Project layout

```
gui/      PySide6 application (detection, signal logic, MQTT publishing)
esp/      ESP32 firmware for the two traffic lights (MQTT + ESP-NOW)
model/    training scripts + the included trained weights
configs/  ready-to-load camera configuration
docs/     installation, usage, and engineering documentation
```
