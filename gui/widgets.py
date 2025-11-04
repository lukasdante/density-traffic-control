
from PySide6.QtWidgets import (QApplication, QMainWindow, QToolBar, QMenuBar, QDialog, QLineEdit, QFormLayout, QVBoxLayout, QHBoxLayout, QPushButton,
                                QMessageBox, QStatusBar, QCheckBox, QGroupBox, QLabel, QFileDialog, QComboBox, QWidget, QSizePolicy, QSplitter,
                                QDockWidget, QTextEdit, QProgressBar, QListWidget, QListWidgetItem, QScrollArea, QSpinBox, QDoubleSpinBox, QSlider,)
from PySide6.QtCore import Qt, QSize, QRegularExpression, QSettings, QByteArray, QTimer, QDateTime, QFileInfo, Signal, QPoint, QRect 
from PySide6.QtGui import QIcon, QAction, QIntValidator, QRegularExpressionValidator, QPixmap, QImage, QPainter, QColor, QPen, QPolygonF
import pyqtgraph as pg
import json
from pathlib import Path
import psutil
import GPUtil
import time
from threading import Thread, Event

from inference import Inferencer
from classes import Camera, LatencyTracker, ProxyCamera

asset_path = str(Path(__file__).parent / "assets")

# TODO: Verify camera first before adding to class if it exists
# TODO: Traffic light logic
# TODO: SORT tracking
# TODO: Camera D lags always....

class MainWindow(QMainWindow):

    # Signals -- defined outside init
    display_settings_changed = Signal(dict)
    experimental_settings_changed = Signal(dict)
    camera_data_changed = Signal(list)

    def __init__(self, app: QApplication):
        super().__init__()
        self.setWindowTitle("Control Panel")
        self.resize(1500, 900)
        self.app = app

        # Qsettings
        self.settings = QSettings("Lanaia Robotics", "TraffIQ-PH")

        # Metrics tracker
        self.latency_tracker = LatencyTracker()

        # Hint docks and load layout
        self.load_layout()
        

        # Menu bar
        menu_bar = MenuBar(self)
        self.setMenuBar(menu_bar)

        # Tool Bar
        main_tool_bar = MainToolBar(self)
        self.addToolBar(Qt.ToolBarArea.LeftToolBarArea, main_tool_bar)
        
        # Status Bar
        status_bar = StatusBar(self)
        self.setStatusBar(status_bar)

        # Cameras
        self.cameras: list[Camera] = []
        self.proxy_cameras: list[ProxyCamera] = []

        # Load all settings and configurations
        self.display_settings = {}
        self.experimental_settings = {}
        self.logs = []
        self.load_all_settings()

        self.start_yolo()
        self._log_message("Initializing YOLO for the first time.", 3000)

    def start_yolo(self):

        # Stop existing YOLO thread
        if hasattr(self, "yolo_thread") and self.yolo_thread.is_alive():
            self.yolo_stop_flag.set()
            self.yolo_thread.join(timeout=2)
            print("[INFO] Previous YOLO thread stopped.")
        
        for lbl in self.cam_labels:
            lbl.clear()
            lbl.setText("Missing camera source")
            lbl.setAlignment(Qt.AlignCenter)

        # Create a new stop flag for the new thread
        self.yolo_stop_flag = Event()

        # Start YOLO again
        if self.experimental_settings.get("use_videos", False):
            target_source = self.proxy_cameras
        else:
            target_source = self.cameras

        self.inferencer = Inferencer(self.cam_labels, target_source, self.yolo_stop_flag,
                                     self.latency_tracker, self.display_settings, self.experimental_settings)
        self.yolo_thread = Thread(
            target=self.inferencer.run,
            daemon=True
        )
        self.display_settings_changed.connect(self.inferencer.on_display_settings_changed)
        self.experimental_settings_changed.connect(self.inferencer.on_experimental_settings_changed)
        self.camera_data_changed.connect(self.inferencer.on_camera_data_changed)

        self.yolo_thread.start()

    def load_layout(self):
        """ Load layout. """
        self.camera_widget = CameraWidget()
        self.metrics_widget = MetricsWidget()
        self.logs_widget = LogsWidget()
        self.chart_left_widget = InferenceMetricsChartWidget()
        self.chart_right_widget = SystemMetricsChartWidget()

        self.cam_labels = [
            self.camera_widget.top_split.widget(0),
            self.camera_widget.top_split.widget(1),
            self.camera_widget.bottom_split.widget(0),
            self.camera_widget.bottom_split.widget(1),
        ]


        # Dock the components
        self.dock_cameras = self._add_dock("Cameras", self.camera_widget, Qt.LeftDockWidgetArea)
        self.dock_metrics = self._add_dock("System Metrics", self.metrics_widget, Qt.RightDockWidgetArea)
        self.dock_logs = self._add_dock("Logs", self.logs_widget, Qt.RightDockWidgetArea)
        self.splitDockWidget(self.dock_metrics, self.dock_logs, Qt.Vertical)
        self.dock_chart_left = self._add_dock("Delay Metrics Chart", self.chart_left_widget, Qt.BottomDockWidgetArea)
        self.dock_chart_right = self._add_dock("System Metrics Chart", self.chart_right_widget, Qt.BottomDockWidgetArea)
        self.splitDockWidget(self.dock_chart_left, self.dock_chart_right, Qt.Horizontal)
        self.resizeDocks([self.dock_cameras, self.dock_chart_left], [700, 200], Qt.Vertical)
    
    def set_layout(self, reset=False):
        """Restore the window layout to its standard configuration."""

        try:
            if reset:
                # Load standard layout if reset
                standard_path = f"{asset_path}/standard_layout.json"
                with open(standard_path, "r") as f:
                    data = json.load(f)
            else:
                # Set the layout using current layout settings
                
                data = self.layout_settings
            
            if not data:
                print("No layout settings can be found.")
                return

            # Restore top-level window geometry and dock state
            self.restoreGeometry(QByteArray.fromHex(data["geometry"].encode()))
            self.restoreState(QByteArray.fromHex(data["state"].encode()))

            # Restore internal camera splitter layout
            if "camera" in data and hasattr(self, "dock_cameras"):
                self.dock_cameras.widget().restore_camera_layout(data["camera"])

            self._log_message("Layout reset to standard configuration.", 3000)

        except FileNotFoundError:
            self._log_message("Standard layout file not found.", 5000)

        except Exception as e:
            print(f"Error loading standard layout: {e}")
            self._log_message("Error resetting layout.", 5000)

    def load_all_settings(self):
        """ Load cameras and UI settings from system store. """

        # Load proxy cameras
        raw_proxies = self.settings.value("proxy_cameras")
        if raw_proxies:
            try:
                data = json.loads(raw_proxies)
                self.proxy_cameras.clear()
                for entry in data:
                    entry.setdefault("direction", "vertical")
                    cam = ProxyCamera(**entry)
                    self.proxy_cameras.append(cam)
                print(f"Loaded {len(self.proxy_cameras)} proxy cameras from the system.")
            except Exception as e:
                print("Error loading cameras:", e)
        
        # Load cameras
        raw_cams = self.settings.value("cameras")
        if raw_cams:
            try:
                data = json.loads(raw_cams)
                self.cameras.clear()
                for entry in data:
                    entry.setdefault("direction", "vertical")
                    cam = Camera(**entry)
                    self.cameras.append(cam)
                print(f"Loaded {len(self.cameras)} cameras from the system.")
            except Exception as e:
                print("Error loading cameras:", e)

        # Load display settings
        raw_display = self.settings.value("display_settings")
        if raw_display:
            try:
                self.display_settings = json.loads(raw_display)
            except Exception:
                self.display_settings = {}
        else:
            # defaults
            self.reset_to_default_settings(settings=['display'])
        
        # ---- Experimental Settings (future-proof)
        raw_experimental = self.settings.value("experimental_settings")
        if raw_experimental:
            try:
                self.experimental_settings = json.loads(raw_experimental)
            except Exception:
                self.experimental_settings = {}
        else:
            self.reset_to_default_settings(settings=['experimental'])
        # Load layout settings
        raw_layout = self.settings.value("layout")
        if raw_layout:
            try:
                self.layout_settings = json.loads(raw_layout)
                self.set_layout()
            except Exception:
                self.layout_settings = {}
        else:
            self.layout_settings = {}

    def reset_to_default_settings(self, _, settings=['display','experimental','layout']):
        """ Resets to default settings. """
        if 'display' in settings:
            self.display_settings = {
                "bounding_boxes": {
                    "firetruck": True,
                    "ambulance": True,
                    "vehicle": True,
                },
                "osd": {
                    "name": True,
                    "location": False,
                    "estimated_congestion": True,
                    "annotations": True,
                    "roi": False,
                    "traffic_phase": True
                },
                "logs": {
                    "log_level": "Info",
                }
            }

        if 'experimental' in settings:
            self.experimental_settings = {
                "use_videos": False,
                "tile_w": 640,
                "tile_h": 360,
                "confidence": 0.30,
            }
        
        if 'layout' in settings:
            self.set_layout(reset=True)
        
        if 'cameras' in settings:
            self.cameras = []
            self.proxy_cameras = []

        self.save_all_settings()
        self.load_all_settings()

    def reset_camera(self):
        self.cameras = []

        self.save_cameras(emit=True)

    def save_all_settings(self):
        """ Save all settings to system. """

        self._save_current_layout()
        
        # Save cameras
        self.save_cameras()

        # Save display settings
        self.settings.setValue("display_settings", json.dumps(self.display_settings))
        
        # Save experimental settings
        self.settings.setValue("experimental_settings", json.dumps(self.experimental_settings))

        # Save geometry and state
        self.settings.setValue("layout", json.dumps(self.layout_settings))

        self._log_message("All settings have been saved in your computer.", 3000)

    def save_cameras(self, emit=False):
        """ Saves camera and proxy camera instances. """
        data = [cam.__dict__.copy() for cam in self.cameras]
        for d in data:
            d.pop("full_link", None)
        self.settings.setValue("cameras", json.dumps(data))

        data = [proxy_cam.__dict__.copy() for proxy_cam in self.proxy_cameras]
        self.settings.setValue("proxy_cameras", json.dumps(data))

        if emit:
            all_cams = self.proxy_cameras if self.experimental_settings.get("use_videos", False) else self.cameras
            self.camera_data_changed.emit(all_cams)

    def _load_configs(self):
        """Import settings from a JSON file."""
        path, _ = QFileDialog.getOpenFileName(self, "Load Configs", "", "JSON Files (*.json)")
        if not path:
            return
        
        with open(path, "r") as f:
            payload = json.load(f)

        # restore other settings
        self.display_settings = payload.get("display_settings", {})
        self.experimental_settings = payload.get("experimental_settings", {})
        
        # restore cameras or proxy cameras
        self.cameras.clear()
        for entry in payload.get("cameras", []):
            self.cameras.append(Camera(**entry))
        
        self.proxy_cameras.clear()
        for entry in payload.get("proxy_cameras", []):
            self.proxy_cameras.append(ProxyCamera(**entry))

        self.start_yolo()
        self._log_message("Starting YOLO with new configs.", 3000)
        


        # restore layout settings
        self.layout_settings = payload.get("layout", {})

        # set layout
        self.set_layout()

        # also update QSettings so it’s consistent
        self.save_all_settings()

        self._log_message(f"Successfully loaded settings from {path}.", 2000)

    def _save_as_configs(self):
        """Export settings to a JSON file."""
        
        path, _ = QFileDialog.getSaveFileName(self, "Save Configs", "", "JSON Files (*.json)")

        self._save_current_layout()
        
        payload = {
            "cameras": [cam.__dict__.copy() for cam in self.cameras],
            "proxy_cameras": [proxy_cam.__dict__.copy() for proxy_cam in self.proxy_cameras],
            "display_settings": self.display_settings,
            "experimental_settings": self.experimental_settings,
            "layout": self.layout_settings,
        }
        for d in payload["cameras"]:
            d.pop("full_link", None)

        with open(path, "w") as f:
            json.dump(payload, f, indent=4)
        
        self._log_message(f"Successfully saved settings to {path}.")

    def _save_current_layout(self):
        self._default_geometry = self.saveGeometry()
        self._default_state = self.saveState()
        self._default_camera_state = self.dock_cameras.widget().save_camera_layout()

        self.layout_settings = {
            "geometry": self._default_geometry.toHex().data().decode(), 
            "state": self._default_state.toHex().data().decode(),
            "camera": self._default_camera_state
        }

    def _quit(self):
        self.app.quit()

    def _add_dock(self, title: str, widget, area):
        dock = QDockWidget(title, self)
        dock.setObjectName(title.replace(" ", "_").lower())
        dock.setWidget(widget)
        dock.setAllowedAreas(
            Qt.LeftDockWidgetArea |
            Qt.RightDockWidgetArea |
            Qt.TopDockWidgetArea |
            Qt.BottomDockWidgetArea
        )
        dock.setFeatures(
            QDockWidget.DockWidgetMovable |
            QDockWidget.DockWidgetFloatable |
            QDockWidget.DockWidgetClosable
        )
        
        self.addDockWidget(area, dock)
        return dock
    
    def _log_message(self, message: str, duration: int):
        self.statusBar().showMessage(message, duration)
        self.logs = [message] + self.logs
        self.dock_logs.widget().setText("\n\n".join(self.logs))
    
    # Overriden

    def resizeEvent(self, event):
        """Force charts to stay smaller relative to cameras after resize/maximize."""
        super().resizeEvent(event)
        self.resizeDocks([self.dock_cameras, self.dock_chart_left], [700, 200], Qt.Vertical)


class MenuBar(QMenuBar):
    def __init__(self, parent: MainWindow=None):
        super().__init__(parent)

        # Add menus
        self.addFileMenu()
        self.addViewMenu()

    def addFileMenu(self):
        self.file_menu = self.addMenu("Settings")
        load_action = self.file_menu.addAction("Load settings")
        save_action = self.file_menu.addAction("Save settings")
        save_as_action = self.file_menu.addAction("Save settings as")
        reset_settings_action = self.file_menu.addAction("Reset to default settings")
        reset_camera_action = self.file_menu.addAction("Reset camera data")
        quit_action = self.file_menu.addAction("Quit")

        load_action.setToolTip("Load settings and configurations.")
        load_action.setStatusTip("Load settings and configurations.")
        save_action.setToolTip("Save settings and configurations.")
        save_action.setStatusTip("Save settings and configurations.")
        save_as_action.setToolTip("Save current settings to a file.")
        save_as_action.setStatusTip("Save current settings to a file.")
        reset_settings_action.setToolTip("Reset settings to default settings except camera data.")
        reset_settings_action.setStatusTip("Reset settings to default settings except camera data.")
        reset_camera_action.setToolTip("Reset camera data.")
        reset_camera_action.setStatusTip("Reset camera data.")
        quit_action.setToolTip("Quit program.")
        quit_action.setStatusTip("Quit program.")

        load_action.triggered.connect(self.parent()._load_configs)
        save_action.triggered.connect(self.parent().save_all_settings)
        save_as_action.triggered.connect(self.parent()._save_as_configs)
        reset_settings_action.triggered.connect(self.parent().reset_to_default_settings)
        reset_camera_action.triggered.connect(self.parent().reset_camera)
        quit_action.triggered.connect(self.parent()._quit)
    
    def addViewMenu(self):
        self.view_menu = self.addMenu("View")

        for dock in [self.parent().dock_cameras,
                     self.parent().dock_metrics,
                     self.parent().dock_logs,
                     self.parent().dock_chart_left,
                     self.parent().dock_chart_right]:
            action = dock.toggleViewAction()
            self.view_menu.addAction(action)
        
        self.view_menu.addSeparator()
        reset_action = self.view_menu.addAction("Reset layout")
        reset_action.triggered.connect(lambda: self.parent().set_layout(reset=True))
        
class MainToolBar(QToolBar):
    def __init__(self, parent):
        super().__init__(parent)
        self.setOrientation(Qt.Orientation.Vertical)
        self.setIconSize(QSize(30, 30))
        self.setWindowTitle("Main Toolbar")
        self.setObjectName("main_toolbar")

        # Add actions
        self.actionAddSource()
        self.actionRemoveSource()
        self.actionChangeDisplaySettings()
        self.actionEditRegion()
        self.actionChangeExperimentalSettings()

    def actionAddSource(self):
        """ Add RTSP source. """
        action = QAction(QIcon(f"{asset_path}/add.png"), "Add", self)
        action.setStatusTip("Add RTSP source.")
        action.setToolTip("Add RTSP source.")
        action.triggered.connect(self._add_source)

        self.addAction(action)
    
    def actionRemoveSource(self):
        """ Remove RTSP source. """
        action = QAction(QIcon(f"{asset_path}/remove.png"), "Remove", self)
        action.setStatusTip("Remove RTSP source.")
        action.setToolTip("Remove RTSP source.")
        action.triggered.connect(self._remove_source)

        self.addAction(action)
    
    def actionChangeDisplaySettings(self):
        """ Change OSD settings. """
        action = QAction(QIcon(f"{asset_path}/display_settings.png"), "Display Settings", self)
        action.setStatusTip("Change display settings.")
        action.setToolTip("Change display settings.")
        action.triggered.connect(self._change_display_settings)

        self.addAction(action)

    def actionEditRegion(self):
        """ Edit region of interest (ROI). """
        action = QAction(QIcon(f"{asset_path}/edit_region.png"), "Edit Region", self)
        action.setStatusTip("Edit region of interest.")
        action.setToolTip("Edit region of interest.")
        action.triggered.connect(self._edit_region)

        self.addAction(action)

    def actionChangeExperimentalSettings(self):
        """ Configure experimental and detection settings. """
        action = QAction(QIcon(f"{asset_path}/experimental.png"), "Experimental Settings", self)
        action.setStatusTip("Change experimental settings.")
        action.setToolTip("Change experimental settings.")
        action.triggered.connect(self._change_experimental_settings)

        self.addAction(action)
    
    # Helper functions

    def _add_source(self):
        use_video = self.parent().experimental_settings.get("use_videos", False)
        if use_video:
            dialog = AddVideoSourceDialog(self)
            if dialog.exec() == QDialog.Accepted:
                data = dialog.get_data()
                proxy_camera = ProxyCamera(**data)
                self.parent().proxy_cameras.append(proxy_camera)
                self.parent().save_cameras()
                self.parent().start_yolo()
                self.parent()._log_message("YOLO restarted with updated camera list.", 3000)

                print(f"Proxy camera '{proxy_camera.name}' added successfully.")
        else:
            dialog = AddSourceDialog(self)
            if dialog.exec() == QDialog.Accepted:
                data = dialog.get_data()
                camera = Camera(**data)
                self.parent().cameras.append(camera) # add current camera to cameras
                self.parent().save_cameras() # save cameras persistently
                self.parent().start_yolo()
                self.parent()._log_message("YOLO restarted with updated camera list.", 3000)
                print(f"Camera added: {camera}")

    def _remove_source(self):
        """Removes selected real and proxy cameras."""
        dialog = RemoveSourceDialog(self)
        if dialog.exec():
            selected = dialog.get_selected_indexes()
            cam_indexes = selected.get("cameras", [])
            proxy_indexes = selected.get("proxies", [])

            if not cam_indexes and not proxy_indexes:
                QMessageBox.information(self, "No Selection", "No sources were selected for removal.")
                return

            # --- Remove from both lists safely (reverse order to avoid index shift) ---
            # Real cameras
            for idx in sorted(cam_indexes, reverse=True):
                if 0 <= idx < len(self.parent().cameras):
                    removed_cam = self.parent().cameras.pop(idx)
                    print(f"[INFO] Removed camera: {removed_cam.name} ({removed_cam.ip_address})")

            # Proxy cameras
            for idx in sorted(proxy_indexes, reverse=True):
                if 0 <= idx < len(self.parent().proxy_cameras):
                    removed_proxy = self.parent().proxy_cameras.pop(idx)
                    print(f"[INFO] Removed proxy camera: {removed_proxy.name} ({removed_proxy.file_path})")

            # --- Save updated lists ---
            self.parent().save_cameras()

            # --- Restart YOLO backend ---
            self.parent().start_yolo()
            self.parent()._log_message("Selected source(s) removed and YOLO restarted.", 3000)


    def _change_display_settings(self):
        dialog = ChangeDisplaySettingsDialog(self.window())
        dialog.set_settings(self.parent().display_settings) # load defaults

        if dialog.exec():
            settings = dialog.get_settings()
            self.parent().display_settings = settings
            self.parent().save_all_settings()
            self.parent()._log_message("Updated display settings", 3000)
            print("User settings:", settings)
        
            self.parent().display_settings_changed.emit(self.parent().display_settings)
        
    def _edit_region(self):
        dlg = EditRegionDialog(self)
        if dlg.exec() == QDialog.Accepted:
            cam = dlg.get_selected_camera()
            if cam:
                self.editor = ROIEditorWindow(cam, parent=self.parent())
                self.editor.show()


    def _change_experimental_settings(self):
        """
        Features:
            2. Checkbox of shown bounding boxes. 
            3. Use other models?
            5. Use videos for demonstration.
        
        """
        dialog = ChangeExperimentalSettingsDialog(self.window())
        dialog.set_settings(self.parent().experimental_settings) # load defaults

        if dialog.exec():
            settings = dialog.get_settings()
            self.parent().experimental_settings = settings
            self.parent().save_all_settings()            
            self.parent()._log_message("Restarting YOLO with updated experimental settings", 3000)
            self.parent().start_yolo()
            print("User settings:", settings)

            self.parent().experimental_settings_changed.emit(self.parent().experimental_settings)

class StatusBar(QStatusBar):
    def __init__(self, parent):
        super().__init__(parent)

class AddSourceDialog(QDialog):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Add Camera Source")
        self.setMinimumWidth(300)

        # Form layout
        layout = QFormLayout()

        self.name_edit = QLineEdit()
        self.address_edit = QLineEdit()
        self.username_edit = QLineEdit()
        self.port_edit = QLineEdit()
        self.channel_edit = QLineEdit()
        self.password_edit = QLineEdit()
        self.location_edit = QLineEdit()
        self.max_cap_edit = QLineEdit()
        self.direction_combo = QComboBox()
        self.direction_combo.addItems(["Vertical", "Horizontal"])
        self.line_edit_fields: list[QLineEdit] = [self.name_edit, self.address_edit, self.username_edit, self.port_edit,
                                 self.channel_edit, self.password_edit, self.location_edit, self.max_cap_edit]

        self.port_edit.setValidator(QIntValidator(1, 65535, self))
        self.max_cap_edit.setValidator(QIntValidator(1, 65535, self))
        self.channel_edit.setValidator(QIntValidator(1, 512, self))
        self.password_edit.setEchoMode(QLineEdit.Password)
        ip_regex = QRegularExpression(
            r"^(25[0-5]|2[0-4]\d|1\d{2}|[1-9]?\d)"
            r"(\.(25[0-5]|2[0-4]\d|1\d{2}|[1-9]?\d)){3}$"
        )
        self.address_edit.setPlaceholderText("e.g. 192.168.0.64")
        self.address_edit.setValidator(QRegularExpressionValidator(ip_regex))
        self.channel_edit.setText("101")
        self.port_edit.setText("554")


        layout.addRow("Camera Name:", self.name_edit)
        layout.addRow("Location:", self.location_edit)
        layout.addRow("IP Address:", self.address_edit)
        layout.addRow("Max cap:", self.max_cap_edit)
        layout.addRow("Username:", self.username_edit)
        layout.addRow("Password:", self.password_edit)
        layout.addRow("RTSP Port:", self.port_edit)
        layout.addRow("Channel:", self.channel_edit)
        layout.addRow("Direction:", self.direction_combo)
        
        

        # Button
        self.add_button = QPushButton("Add")
        self.add_button.clicked.connect(self.validate_and_accept)

        vbox = QVBoxLayout()
        vbox.addLayout(layout)
        vbox.addWidget(self.add_button)

        self.setLayout(vbox)
    
    def validate_and_accept(self):
        """ Validates the form and checks if all fields are populated. """
        if not all([field.text().strip() for field in self.line_edit_fields]):
            QMessageBox.warning(self, "Missing Data", "Please fill all fields!")
            return

        self.accept()
    
    def get_data(self):
        """Package dialog data into a dict (or return Camera directly)."""
        return {
            "name": self.name_edit.text().strip(),
            "ip_address": self.address_edit.text().strip(),
            "username": self.username_edit.text().strip(),
            "password": self.password_edit.text().strip(),
            "rtsp_port": int(self.port_edit.text().strip()),
            "channel": int(self.channel_edit.text().strip()),
            "location": self.location_edit.text().strip(),
            "max_cap": self.max_cap_edit.text().strip(),
            "direction": self.direction_combo.currentText().lower(),
        }
    
class RemoveSourceDialog(QDialog):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Remove Camera Source(s)")
        self.setMinimumWidth(800)
        self.parent_window = parent.parent()  # reference to MainWindow

        main_layout = QVBoxLayout(self)

        # ==============================
        # --- Two Columns (GroupBoxes) ---
        # ==============================
        columns_layout = QHBoxLayout()

        # --- Camera Sources GroupBox ---
        self.camera_group = QGroupBox("Camera Sources")
        cam_layout = QVBoxLayout(self.camera_group)
        self.camera_list = QListWidget()
        self.camera_list.setSelectionMode(QListWidget.MultiSelection)
        self.camera_list.setAlternatingRowColors(True)
        cam_layout.addWidget(self.camera_list)

        # populate camera list
        for i, cam in enumerate(self.parent_window.cameras):
            item_text = f"{i+1}. {cam.name} — {cam.ip_address} ({cam.location})"
            item = QListWidgetItem(item_text)
            item.setFlags(item.flags() | Qt.ItemIsUserCheckable)
            item.setCheckState(Qt.Unchecked)
            self.camera_list.addItem(item)

        # --- Proxy Camera Sources GroupBox ---
        self.proxy_group = QGroupBox("Proxy Camera Sources")
        proxy_layout = QVBoxLayout(self.proxy_group)
        self.proxy_list = QListWidget()
        self.proxy_list.setSelectionMode(QListWidget.MultiSelection)
        self.proxy_list.setAlternatingRowColors(True)
        proxy_layout.addWidget(self.proxy_list)

        # populate proxy list
        for i, proxy in enumerate(self.parent_window.proxy_cameras):
            item_text = f"{i+1}. {proxy.name} — {proxy.file_path} ({proxy.location})"
            item = QListWidgetItem(item_text)
            item.setFlags(item.flags() | Qt.ItemIsUserCheckable)
            item.setCheckState(Qt.Unchecked)
            self.proxy_list.addItem(item)

        # --- Make both scrollable ---
        cam_scroll = QScrollArea()
        cam_scroll.setWidgetResizable(True)
        cam_scroll.setWidget(self.camera_group)

        proxy_scroll = QScrollArea()
        proxy_scroll.setWidgetResizable(True)
        proxy_scroll.setWidget(self.proxy_group)

        # --- Add to horizontal layout ---
        columns_layout.addWidget(cam_scroll)
        columns_layout.addWidget(proxy_scroll)
        main_layout.addLayout(columns_layout)

        # ==============================
        # --- Buttons ---
        # ==============================
        btn_layout = QHBoxLayout()
        self.remove_btn = QPushButton("Remove Selected")
        self.cancel_btn = QPushButton("Cancel")
        btn_layout.addWidget(self.remove_btn)
        btn_layout.addWidget(self.cancel_btn)
        main_layout.addLayout(btn_layout)

        # --- Connections ---
        self.remove_btn.clicked.connect(self.accept)
        self.cancel_btn.clicked.connect(self.reject)


    def get_selected_indexes(self):
        """Return dict of selected (checked) cameras and proxy cameras."""
        selected = {"cameras": [], "proxies": []}

        # regular cameras
        for i in range(self.camera_list.count()):
            item = self.camera_list.item(i)
            if item.checkState() == Qt.Checked:
                selected["cameras"].append(i)

        # proxy cameras
        for i in range(self.proxy_list.count()):
            item = self.proxy_list.item(i)
            if item.checkState() == Qt.Checked:
                selected["proxies"].append(i)

        return selected

class AddVideoSourceDialog(QDialog):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Add Proxy Video Source")
        self.setMinimumWidth(400)
        self.selected_proxy = None  # will hold the created ProxyCamera

        # --- Layout setup ---
        main_layout = QVBoxLayout()
        form_layout = QFormLayout()
        form_layout.setSpacing(10)

        # Info label
        info_label = QLabel(
            "The 'Use Videos' setting is enabled.\n"
            "Fill in the details below to register a proxy camera."
        )
        info_label.setWordWrap(True)
        main_layout.addWidget(info_label)

        # Name field
        self.name_edit = QLineEdit()
        form_layout.addRow("Camera Name:", self.name_edit)

        # Location field
        self.location_edit = QLineEdit()
        form_layout.addRow("Location:", self.location_edit)

        # Max cap field
        self.max_cap_edit = QLineEdit()
        form_layout.addRow("Max cap:", self.max_cap_edit)

        # Direction dropdown
        self.direction_combo = QComboBox()
        self.direction_combo.addItems(["Vertical", "Horizontal"])
        form_layout.addRow("Direction:", self.direction_combo)

        # File picker
        file_layout = QHBoxLayout()
        self.file_edit = QLineEdit()
        self.file_edit.setReadOnly(True)
        self.browse_btn = QPushButton("Browse")
        self.browse_btn.clicked.connect(self.load_video)
        file_layout.addWidget(self.file_edit)
        file_layout.addWidget(self.browse_btn)
        form_layout.addRow("Video File:", file_layout)

        self.line_edit_fields = [self.name_edit, self.location_edit, self.file_edit, self.max_cap_edit]

        main_layout.addLayout(form_layout)

        # Buttons
        self.submit_btn = QPushButton("Add Proxy Camera")
        self.submit_btn.clicked.connect(self.validate_and_accept)
        main_layout.addWidget(self.submit_btn, alignment=Qt.AlignRight)

        self.setLayout(main_layout)

    # ------------------------------------------------------
    def load_video(self):
        """Open file dialog to select a video."""
        file_path, _ = QFileDialog.getOpenFileName(
            self,
            "Select a Video File",
            "",
            "Video Files (*.mp4 *.avi *.mkv *.mov)"
        )
        if file_path:
            self.file_edit.setText(file_path)
            # Auto-suggest name and RTSP location if empty
            base_name = QFileInfo(file_path).baseName()
            if not self.name_edit.text():
                self.name_edit.setText(f"Proxy-{base_name}")
            if not self.location_edit.text():
                self.location_edit.setText(f"rtsp://127.0.0.1:8554/{base_name}")

    def validate_and_accept(self):
        """ Validates the form and checks if all fields are populated. """
        if not all([field.text().strip() for field in self.line_edit_fields]):
            QMessageBox.warning(self, "Missing Data", "Please fill all fields!")
            return

        self.accept()
    
    def get_data(self):
        """Package dialog data into a dict (or return Camera directly)."""
        return {
            "name": self.name_edit.text().strip(),
            "location": self.location_edit.text().strip(),
            "file_path": self.file_edit.text().strip(),
            "max_cap": self.max_cap_edit.text().strip(),
            "direction": self.direction_combo.currentText().lower(),
        }

class ChangeDisplaySettingsDialog(QDialog):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Change Display Settings")
        self.setMinimumWidth(400)

        main_layout = QVBoxLayout()

        # --- Group 1: Bounding Boxes ---
        bbox_group = QGroupBox("Bounding Box Display")
        bbox_layout = QVBoxLayout()
        bbox_layout.addWidget(QLabel("Choose which bounding boxes to show:"))

        self.cb_firetruck = QCheckBox("Firetrucks")
        self.cb_ambulance = QCheckBox("Ambulances")
        self.cb_vehicle = QCheckBox("Vehicles")

        bbox_layout.addWidget(self.cb_firetruck)
        bbox_layout.addWidget(self.cb_ambulance)
        bbox_layout.addWidget(self.cb_vehicle)

        bbox_group.setLayout(bbox_layout)
        main_layout.addWidget(bbox_group)

        # --- Group 2: OSD (On-Screen Display) ---
        osd_group = QGroupBox("On-Screen Display (OSD)")
        osd_layout = QVBoxLayout()
        osd_layout.addWidget(QLabel("Choose which information to overlay:"))

        self.cb_name = QCheckBox("Name")
        self.cb_location = QCheckBox("Location")
        self.cb_congestion = QCheckBox("Estimated Congestion")
        self.cb_annotations = QCheckBox("Annotations")
        self.cb_roi = QCheckBox("Region of Interests")
        self.cb_traffic_phase = QCheckBox("Traffic Phase")

        osd_layout.addWidget(self.cb_name)
        osd_layout.addWidget(self.cb_location)
        osd_layout.addWidget(self.cb_congestion)
        osd_layout.addWidget(self.cb_annotations)
        osd_layout.addWidget(self.cb_roi)
        osd_layout.addWidget(self.cb_traffic_phase)

        osd_group.setLayout(osd_layout)
        main_layout.addWidget(osd_group)

        # --- Group 3: Logs
        log_group = QGroupBox("Logs")
        log_layout = QVBoxLayout()
        log_layout.addWidget(QLabel("Configure logs settings:"))
        self.cb_log_level = QComboBox()
        self.cb_log_level.addItems(["Debug", "Info" ,"Errors"])

        log_layout.addWidget(self.cb_log_level)

        log_group.setLayout(log_layout)
        main_layout.addWidget(log_group)

        # --- Buttons ---
        button_layout = QHBoxLayout()
        self.ok_button = QPushButton("Save")
        self.cancel_button = QPushButton("Cancel")

        self.ok_button.clicked.connect(self.accept)
        self.cancel_button.clicked.connect(self.reject)

        button_layout.addWidget(self.ok_button)
        button_layout.addWidget(self.cancel_button)
        main_layout.addLayout(button_layout)

        self.setLayout(main_layout)

    def get_settings(self):
        """Return the chosen settings as a dict"""
        return {
            "bounding_boxes": {
                "firetruck": self.cb_firetruck.isChecked(),
                "ambulance": self.cb_ambulance.isChecked(),
                "vehicle": self.cb_vehicle.isChecked(),
            },
            "osd": {
                "name": self.cb_name.isChecked(),
                "location": self.cb_location.isChecked(),
                "estimated_congestion": self.cb_congestion.isChecked(),
                "annotations": self.cb_annotations.isChecked(),
                "roi": self.cb_roi.isChecked(),
                "traffic_phase": self.cb_traffic_phase.isChecked()
            },
            "logs": {
                "log_level": self.cb_log_level.currentText()
            }
        }
    
    def set_settings(self, settings: dict):
        bb = settings.get("bounding_boxes", {})
        osd = settings.get("osd", {})
        logs = settings.get("logs", {})

        self.cb_firetruck.setChecked(bb.get("firetruck", False))
        self.cb_ambulance.setChecked(bb.get("ambulance", False))
        self.cb_vehicle.setChecked(bb.get("vehicle", False))

        self.cb_name.setChecked(osd.get("name", False))
        self.cb_location.setChecked(osd.get("location", False))
        self.cb_congestion.setChecked(osd.get("estimated_congestion", False))
        self.cb_annotations.setChecked(osd.get("annotations", False))
        self.cb_roi.setChecked(osd.get("roi", False))
        self.cb_traffic_phase.setChecked(osd.get("traffic_phase", False))
        log_level = logs.get("log_level", "Info")   # default to "Info"
        idx = self.cb_log_level.findText(log_level)
        if idx >= 0:
            self.cb_log_level.setCurrentIndex(idx)

class EditRegionDialog(QDialog):
    """Dialog to select which camera's ROI to edit."""
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Select Camera to Edit ROI")
        self.setMinimumWidth(400)

        if self.parent().parent().experimental_settings.get("use_videos", False):
            self.cameras = self.parent().parent().proxy_cameras
        else:
            self.cameras = self.parent().parent().cameras

        layout = QVBoxLayout(self)
        layout.addWidget(QLabel("Choose a camera:"))

        self.combo = QComboBox()
        for cam in self.cameras:
            self.combo.addItem(cam.name)
        layout.addWidget(self.combo)

        btn_layout = QHBoxLayout()
        ok_btn = QPushButton("OK")
        cancel_btn = QPushButton("Cancel")
        btn_layout.addWidget(ok_btn)
        btn_layout.addWidget(cancel_btn)
        layout.addLayout(btn_layout)

        ok_btn.clicked.connect(self.accept)
        cancel_btn.clicked.connect(self.reject)

    def get_selected_camera(self):
        idx = self.combo.currentIndex()
        return self.cameras[idx] if 0 <= idx < len(self.cameras) else None

class ROIEditorWindow(QMainWindow):
    """ROI editor window showing a snapshot and an adjustable quadrilateral."""
    def __init__(self, camera, parent=None):
        super().__init__(parent)
        self.camera = camera
        self.setWindowTitle(f"Edit ROI — {camera.name}")

        # --- Load snapshot first ---
        snapshot, frame_rect = self._get_snapshot()

        # --- Central widget setup ---
        central = QWidget()
        self.setCentralWidget(central)
        layout = QVBoxLayout(central)
        layout.setContentsMargins(10, 10, 10, 10)
        layout.setSpacing(6)

        # --- Image label ---
        self.image_label = QLabel()
        self.image_label.setPixmap(snapshot)
        self.image_label.setAlignment(Qt.AlignCenter)
        self.image_label.setStyleSheet("background: black;")
        layout.addWidget(self.image_label)

        # --- ROI overlay ---
        self.overlay = ROICanvas(self.image_label, camera.roi, frame_rect)
        self.overlay.setGeometry(frame_rect)
        self.overlay.show()

        # --- Buttons ---
        btn_layout = QHBoxLayout()
        save_btn = QPushButton("Save")
        cancel_btn = QPushButton("Cancel")
        btn_layout.addStretch()
        btn_layout.addWidget(save_btn)
        btn_layout.addWidget(cancel_btn)
        layout.addLayout(btn_layout)

        save_btn.clicked.connect(self._save_roi)
        cancel_btn.clicked.connect(self.close)

        # --- Let the window adapt automatically ---
        self.adjustSize()


    def _get_snapshot(self):
        """Capture a single frame and scale to fit 1280x720 while keeping aspect ratio."""
        import cv2, numpy as np

        target_w, target_h = 1280, 720
        frame_rect = QRect(0, 0, target_w, target_h)

        try:
            if hasattr(self.camera, "file_path"):
                src = self.camera.file_path
            elif hasattr(self.camera, "full_link"):
                src = self.camera.full_link
            else:
                raise AttributeError("Camera source not found.")

            cap = cv2.VideoCapture(src)
            ret, frame = cap.read()
            cap.release()

            if not ret or frame is None:
                raise RuntimeError("Failed to capture frame.")

            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            h, w, _ = frame.shape

            scale = min(target_w / w, target_h / h)
            new_w, new_h = int(w * scale), int(h * scale)
            frame = cv2.resize(frame, (new_w, new_h), interpolation=cv2.INTER_AREA)

            canvas = np.zeros((target_h, target_w, 3), dtype=np.uint8)
            x_offset = (target_w - new_w) // 2
            y_offset = (target_h - new_h) // 2
            canvas[y_offset:y_offset + new_h, x_offset:x_offset + new_w] = frame

            frame_rect = QRect(x_offset, y_offset, new_w, new_h)

            qimg = QImage(canvas.data, target_w, target_h, target_w * 3, QImage.Format_RGB888)
            pixmap = QPixmap.fromImage(qimg)
            return pixmap, frame_rect

        except Exception as e:
            print(f"[WARN] Snapshot failed: {e}")
            pix = QPixmap(target_w, target_h)
            pix.fill(Qt.black)
            return pix, frame_rect


    def _save_roi(self):
        if not hasattr(self.overlay, "points"):
            return
        pix_rect = self.image_label.pixmap().rect()
        w, h = pix_rect.width(), pix_rect.height()
        normalized = [(p.x() / w, p.y() / h) for p in self.overlay.points]
        self.camera.roi = normalized
        print(f"[INFO] Saved ROI for {self.camera.name}: {normalized}")
        self.parent().save_cameras(emit=True)
        self.close()

class ROICanvas(QWidget):
    """A direct, embedded ROI polygon editor (bounded to frame_rect)."""
    def __init__(self, parent, roi_points=None, frame_rect=None):
        super().__init__(parent)
        self.frame_rect = frame_rect or parent.pixmap().rect()
        self.setAttribute(Qt.WA_TranslucentBackground, True)
        self.setAttribute(Qt.WA_TransparentForMouseEvents, False)
        self.dragging = False
        self.active_handle = None
        self.handle_size = 10

        w, h = self.frame_rect.width(), self.frame_rect.height()
        x0, y0 = self.frame_rect.x(), self.frame_rect.y()

        if roi_points:
            # Restore saved ROI in normalized coordinates
            self.points = [QPoint(int(x0 + x * w), int(y0 + y * h)) for x, y in roi_points]
        else:
            scale = 0.99
            pad_x = int((1 - scale) * w / 2)
            pad_y = int((1 - scale) * h / 2)

            self.points = [
                QPoint(x0 + pad_x, y0 + pad_y),
                QPoint(x0 + w - pad_x, y0 + pad_y),
                QPoint(x0 + w - pad_x, y0 + h - pad_y),
                QPoint(x0 + pad_x, y0 + h - pad_y)
            ]

    # --- painting ---
    def paintEvent(self, event):
        p = QPainter(self)
        p.setRenderHint(QPainter.Antialiasing)
        poly = QPolygonF(self.points)
        p.setBrush(QColor(0, 0, 255, 80))
        p.setPen(QPen(QColor("white"), 2))
        p.drawPolygon(poly)

        handle_color = QColor(173, 216, 230)
        for pt in self.points:
            p.fillRect(QRect(pt.x() - 5, pt.y() - 5, 10, 10), handle_color)

    # --- interaction ---
    def mousePressEvent(self, e):
        pos = e.pos()
        for i, pt in enumerate(self.points):
            if QRect(pt.x() - 5, pt.y() - 5, 10, 10).contains(pos):
                self.active_handle = i
                self.dragging = True
                return
        if QPolygonF(self.points).containsPoint(pos, Qt.OddEvenFill):
            self.active_handle = -1
            self.drag_offset = pos
            self.dragging = True

    def mouseMoveEvent(self, e):
        if not self.dragging:
            return

        pos = e.pos()
        fr = self.frame_rect

        if self.active_handle == -1:
            # Move entire polygon
            dx = pos.x() - self.drag_offset.x()
            dy = pos.y() - self.drag_offset.y()
            moved_points = [QPoint(pt.x() + dx, pt.y() + dy) for pt in self.points]

            # Clamp polygon within frame bounds
            poly_rect = QPolygonF(moved_points).boundingRect()
            if poly_rect.left() < fr.left():
                dx += fr.left() - poly_rect.left()
            if poly_rect.top() < fr.top():
                dy += fr.top() - poly_rect.top()
            if poly_rect.right() > fr.right():
                dx -= poly_rect.right() - fr.right()
            if poly_rect.bottom() > fr.bottom():
                dy -= poly_rect.bottom() - fr.bottom()

            self.points = [QPoint(pt.x() + dx, pt.y() + dy) for pt in self.points]
            self.drag_offset = pos

        else:
            # Move a single handle (bounded)
            x = max(fr.left(), min(pos.x(), fr.right()))
            y = max(fr.top(), min(pos.y(), fr.bottom()))
            self.points[self.active_handle] = QPoint(x, y)

        self.update()

    def mouseReleaseEvent(self, e):
        self.dragging = False
        self.active_handle = None


class ChangeExperimentalSettingsDialog(QDialog):
    """
        1. Use videos for demonstration.
        2. Adjust tile width and height.
    """
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Change Experimental Settings")
        self.setMinimumWidth(400)

        main_layout = QVBoxLayout()

        # Group
        settings_group = QGroupBox("Experimental Settings")
        settings_layout = QVBoxLayout()
        settings_layout.addWidget(QLabel("Configure experimental settings:"))

        self.cb_use_videos = QCheckBox("Use videos instead for demonstration purposes.")
        settings_layout.addWidget(self.cb_use_videos)

        # --- Tile Settings ---
        settings_layout.addSpacing(10)
        tile_layout = QVBoxLayout()

        tile_layout.addWidget(QLabel("Tile Width:"))
        self.set_tile_w = QSpinBox()
        self.set_tile_w.setRange(640, 3840)
        self.set_tile_w.setValue(1280)
        tile_layout.addWidget(self.set_tile_w)

        tile_layout.addWidget(QLabel("Tile Height:"))
        self.set_tile_h = QSpinBox()
        self.set_tile_h.setRange(360, 2160)
        self.set_tile_h.setValue(720)
        tile_layout.addWidget(self.set_tile_h)
        tile_layout.addSpacing(10)


        # Inside your layout setup
        confidence_layout = QHBoxLayout()
        confidence_layout.addWidget(QLabel("Model confidence:"))

        # Slider setup (0–100 mapped to 0.0–1.0)
        self.slider_conf = QSlider(Qt.Horizontal)
        self.slider_conf.setRange(0, 100)
        self.slider_conf.setValue(30)
        confidence_layout.addWidget(self.slider_conf)

        # Double spinbox setup (0.00–1.00)
        self.spin_conf = QDoubleSpinBox()
        self.spin_conf.setRange(0.0, 1.0)
        self.spin_conf.setDecimals(2)
        self.spin_conf.setSingleStep(0.01)
        self.spin_conf.setValue(0.30)
        confidence_layout.addWidget(self.spin_conf)

        # --- Synchronize both widgets ---
        self.slider_conf.valueChanged.connect(
            lambda v: self.spin_conf.setValue(v / 100)
        )
        self.spin_conf.valueChanged.connect(
            lambda v: self.slider_conf.setValue(int(v * 100))
        )

        tile_layout.addLayout(confidence_layout)

        settings_layout.addLayout(tile_layout)

        settings_group.setLayout(settings_layout)
        main_layout.addWidget(settings_group)

        # --- Buttons ---
        button_layout = QHBoxLayout()
        self.ok_button = QPushButton("Save")
        self.cancel_button = QPushButton("Cancel")

        self.ok_button.clicked.connect(self.accept)
        self.cancel_button.clicked.connect(self.reject)

        button_layout.addWidget(self.ok_button)
        button_layout.addWidget(self.cancel_button)
        main_layout.addLayout(button_layout)

        self.setLayout(main_layout)

    def get_settings(self):
        """Return the chosen settings as a dict"""
        return {
            "use_videos": self.cb_use_videos.isChecked(),
            "tile_w": self.set_tile_w.value(),
            "tile_h": self.set_tile_h.value(),
            "confidence": self.spin_conf.value(),
        }
    
    def set_settings(self, settings: dict):
        # back here
        self.cb_use_videos.setChecked(settings.get("use_videos", False))
        self.set_tile_w.setValue(settings.get("tile_w", 640))
        self.set_tile_h.setValue(settings.get("tile_h", 360))
        self.spin_conf.setValue(settings.get("confidence", 0.30))

class CameraWidget(QSplitter):
    def __init__(self, parent=None):
        super().__init__(parent)

        cam1 = self.make_camera_widget("Missing camera source")
        cam2 = self.make_camera_widget("Missing camera source")
        cam3 = self.make_camera_widget("Missing camera source")
        cam4 = self.make_camera_widget("Missing camera source")

        # top row (cam1 | cam2)
        self.top_split = QSplitter(Qt.Horizontal)
        self.top_split.addWidget(cam1)
        self.top_split.addWidget(cam2)

        # bottom row (cam3 | cam4)
        self.bottom_split = QSplitter(Qt.Horizontal)
        self.bottom_split.addWidget(cam3)
        self.bottom_split.addWidget(cam4)

        # vertical split (top row over bottom row)
        self.setOrientation(Qt.Vertical)
        self.addWidget(self.top_split)
        self.addWidget(self.bottom_split)
    
    def make_camera_widget(self, text):
        lbl = QLabel(text)
        lbl.setAlignment(Qt.AlignCenter)
        lbl.setStyleSheet("background: #111111; color: white; border: 1px solid black;")
        lbl.setSizePolicy(QSizePolicy.Ignored, QSizePolicy.Ignored)
        lbl.setScaledContents(False)
        return lbl

    def save_camera_layout(self):
        """Return all splitter states as dict."""
        return {
            "main": bytes(self.saveState()).hex(),
            "top": bytes(self.top_split.saveState()).hex(),
            "bottom": bytes(self.bottom_split.saveState()).hex(),
        }

    def restore_camera_layout(self, data: dict):
        """Restore all splitter geometries."""
        if not data:
            return
        if "main" in data:
            self.restoreState(bytes.fromhex(data["main"]))
        if "top" in data:
            self.top_split.restoreState(bytes.fromhex(data["top"]))
        if "bottom" in data:
            self.bottom_split.restoreState(bytes.fromhex(data["bottom"]))

class MetricsWidget(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("System Metrics")
        self.setFixedSize(250, 250)

        layout = QFormLayout()
        layout.setVerticalSpacing(4)
        layout.setHorizontalSpacing(10)

        self.date_label = QLabel()
        self.date_label.setAlignment(Qt.AlignCenter)
        self.date_label.setStyleSheet("font-weight: bold; font-size: 13px;")

        self.time_label = QLabel()
        self.time_label.setAlignment(Qt.AlignCenter)
        self.time_label.setStyleSheet("font-weight: bold; font-size: 13px;")
        

        self.cpu_bar = QProgressBar()
        self.cpu_bar.setRange(0, 100)
        self.cpu_bar.setFormat("%p%")          # show percentage
        self.cpu_bar.setAlignment(Qt.AlignCenter)
        self.cpu_bar.setTextVisible(True)

        self.ram_bar = QProgressBar()
        self.ram_bar.setRange(0, 100)
        self.ram_bar.setFormat("%p%")          # show percentage
        self.ram_bar.setAlignment(Qt.AlignCenter)
        self.ram_bar.setTextVisible(True)

        self.gpu_bar = QProgressBar()
        self.gpu_bar.setRange(0, 100)
        self.gpu_bar.setFormat("%p%")          # show percentage
        self.gpu_bar.setAlignment(Qt.AlignCenter)
        self.gpu_bar.setTextVisible(True)

        self.uptime_label = QLabel("-- s")
        self.fps_label = QLabel("--")
        self.latency_total = QLabel("-- ms")
        self.latency_camera = QLabel("-- ms")
        self.latency_inference = QLabel("-- ms")
        self.latency_display = QLabel("-- ms")

        layout.addRow(self.date_label)
        layout.addRow(self.time_label)
        layout.addRow(QLabel("CPU:"), self.cpu_bar)
        layout.addRow(QLabel("RAM:"), self.ram_bar)
        layout.addRow(QLabel("GPU:"), self.gpu_bar)
        layout.addRow(QLabel("Uptime:"), self.uptime_label)
        layout.addRow(QLabel("FPS:"), self.fps_label)
        layout.addRow(QLabel("Total Delay:"), self.latency_total)
        layout.addRow(QLabel("Cam Delay:"), self.latency_camera)
        layout.addRow(QLabel("Inf Delay:"), self.latency_inference)
        layout.addRow(QLabel("Disp Delay:"), self.latency_display)

        self.setLayout(layout)

        psutil.cpu_percent(interval=None)

        self.start_time = time.time()
        self.timer = QTimer(self)
        self.timer.timeout.connect(self.update_metrics)
        self.timer.start(1000)  # update every second

    def update_metrics(self):
        cpu = psutil.cpu_percent(interval=None)
        self.cpu_bar.setValue(int(cpu))

        ram = psutil.virtual_memory().percent
        self.ram_bar.setValue(int(ram))

        try:
            gpus = GPUtil.getGPUs()
            gpu_usage = gpus[0].load * 100 if gpus else 0
        except Exception:
            gpu_usage = 0
        self.gpu_bar.setValue(int(gpu_usage))

        uptime = int(time.time() - self.start_time)
        self.uptime_label.setText(f"{uptime} s")

        current = QDateTime.currentDateTime()
        self.date_label.setText(current.toString("dddd, MMMM dd yyyy"))
        self.time_label.setText(current.toString("hh:mm:ss ap"))

        summary = self.parent().parent().latency_tracker.summary()
        
        if summary["total"] > 0:
            fps = 1.0 / summary["total"]     # correct FPS computation
            self.fps_label.setText(f"{fps:.1f}")
        self.latency_total.setText(f"{summary['total']*1000:.1f} ms")
        self.latency_camera.setText(f"{summary['capture']*1000:.1f} ms")
        self.latency_inference.setText(f"{summary['inference']*1000:.1f} ms")
        self.latency_display.setText(f"{summary['display']*1000:.1f} ms")


class LogsWidget(QTextEdit):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setReadOnly(True)
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)

class SystemMetricsChartWidget(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)

        layout = QVBoxLayout(self)
        self.plot_widget = pg.PlotWidget(background="#222")
        layout.addWidget(self.plot_widget)

        # --- Plot setup ---
        self.plot_widget.showGrid(x=True, y=True, alpha=0.3)
        self.plot_widget.setLabel("bottom", "Time", units="s")
        self.plot_widget.setLabel("left", "Usage", units="%")
        self.plot_widget.setYRange(0, 100)
        self.plot_widget.addLegend()

        # Lines for CPU, RAM, GPU
        self.cpu_line = self.plot_widget.plot(pen=pg.mkPen("y", width=2), name="CPU")
        self.ram_line = self.plot_widget.plot(pen=pg.mkPen("r", width=2), name="RAM")
        self.gpu_line = self.plot_widget.plot(pen=pg.mkPen("c", width=2), name="GPU")

        # Buffers (180 data points ≈ 3 min at 1 Hz)
        self.max_points = 120
        self.cpu_data, self.ram_data, self.gpu_data = [], [], []
        self.time_data = []

        # Prime psutil CPU measurement
        psutil.cpu_percent(interval=None)

        # Timer for updates
        self.start_time = time.time()
        self.timer = QTimer(self)
        self.timer.timeout.connect(self.update_data)
        self.timer.start(1000)  # every 1 s

    def update_data(self):
        now = time.time() - self.start_time
        cpu = psutil.cpu_percent(interval=None)
        ram = psutil.virtual_memory().percent

        try:
            gpus = GPUtil.getGPUs()
            gpu = gpus[0].load * 100 if gpus else 0
        except Exception:
            gpu = 0

        # Append new samples
        self.time_data.append(now)
        self.cpu_data.append(cpu)
        self.ram_data.append(ram)
        self.gpu_data.append(gpu)

        # Keep only last 180 samples
        if len(self.time_data) > self.max_points:
            self.time_data = self.time_data[-self.max_points:]
            self.cpu_data = self.cpu_data[-self.max_points:]
            self.ram_data = self.ram_data[-self.max_points:]
            self.gpu_data = self.gpu_data[-self.max_points:]

        # Update plot lines
        self.cpu_line.setData(self.time_data, self.cpu_data)
        self.ram_line.setData(self.time_data, self.ram_data)
        self.gpu_line.setData(self.time_data, self.gpu_data)

class InferenceMetricsChartWidget(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)

        layout = QVBoxLayout(self)
        self.plot_widget = pg.PlotWidget(background="#222")
        layout.addWidget(self.plot_widget)

        # --- Plot setup ---
        self.plot_widget.showGrid(x=True, y=True, alpha=0.3)
        self.plot_widget.setLabel("bottom", "Frame")
        self.plot_widget.setLabel("left", "Latency", units="ms")
        self.plot_widget.enableAutoRange(axis='y', enable=True)
        self.plot_widget.addLegend()

        # Lines for CPU, RAM, GPU
        self.cam_line = self.plot_widget.plot(pen=pg.mkPen("y", width=2), name="Camera")
        self.inf_line = self.plot_widget.plot(pen=pg.mkPen("r", width=2), name="Inference")
        self.dis_line = self.plot_widget.plot(pen=pg.mkPen("c", width=2), name="Display")

        # Buffers (180 data points ≈ 3 min at 1 Hz)
        self.max_points = 120
        self.cam_data, self.inf_data, self.dis_data = [], [], []
        self.time_data = []

        # Timer for updates
        self.start_time = time.time()
        self.timer = QTimer(self)
        self.timer.timeout.connect(self.update_data)
        self.timer.start(1000)  # every 1 s

    def update_data(self):
        now = time.time() - self.start_time
        data = self.parent().parent().latency_tracker.summary()

        cam = data.get("capture", 0) * 1000
        inf = data.get("inference", 0) * 1000
        dis = data.get("display", 0) * 1000

        # Append new samples
        self.time_data.append(now)
        self.cam_data.append(cam)
        self.inf_data.append(inf)
        self.dis_data.append(dis)

        # Keep only last 180 samples
        if len(self.time_data) > self.max_points:
            self.time_data = self.time_data[-self.max_points:]
            self.cam_data = self.cam_data[-self.max_points:]
            self.inf_data = self.inf_data[-self.max_points:]
            self.dis_data = self.dis_data[-self.max_points:]

        # Update plot lines
        self.cam_line.setData(self.time_data, self.cam_data)
        self.inf_line.setData(self.time_data, self.inf_data)
        self.dis_line.setData(self.time_data, self.dis_data)
