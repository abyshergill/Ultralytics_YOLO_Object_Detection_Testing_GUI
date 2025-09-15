import cv2
import numpy as np
from PyQt5.QtWidgets import (QMainWindow, QWidget, QVBoxLayout, 
                           QHBoxLayout, QGridLayout, QLabel, QPushButton, 
                           QSlider, QFileDialog, QMessageBox, QGroupBox,
                           QCheckBox, QProgressBar, QTextEdit, QSplitter)
from PyQt5.QtCore import Qt, QTimer
from PyQt5.QtGui import QPixmap, QImage, QFont
from ultralytics import YOLO
import os

from .videothread import VideoThread

class YOLODetectionGUI(QMainWindow):
    def __init__(self):
        super().__init__()
        self.model = None
        self.current_image = None
        self.video_thread = VideoThread()
        
        self.video_thread.frame_ready.connect(self.update_display)
        self.video_thread.detection_stats.connect(self.update_stats)
        self.video_thread.finished.connect(self.on_detection_finished)
        
        self.init_ui()
        
    def init_ui(self):
        self.setWindowTitle("YOLO Object Detection GUI Using PyQt5 By Aby")
        self.setGeometry(100, 100, 1200, 800)
        
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        main_layout = QHBoxLayout(central_widget)
        
        splitter = QSplitter(Qt.Horizontal)
        main_layout.addWidget(splitter)
        
        left_panel = QWidget()
        left_panel.setMaximumWidth(350)
        left_panel.setMinimumWidth(300)
        left_layout = QVBoxLayout(left_panel)
        
        right_panel = QWidget()
        right_layout = QVBoxLayout(right_panel)
        
        splitter.addWidget(left_panel)
        splitter.addWidget(right_panel)
        splitter.setSizes([350, 850])
        
        self.setup_controls(left_layout)
        
        self.setup_display(right_layout)
        
    def setup_controls(self, layout):
        model_group = QGroupBox("Model Selection")
        model_layout = QVBoxLayout()
        
        self.model_path_label = QLabel("No model selected")
        self.model_path_label.setWordWrap(True)
        model_layout.addWidget(self.model_path_label)
        
        model_btn_layout = QHBoxLayout()
        self.browse_model_btn = QPushButton("Browse Model")
        self.load_model_btn = QPushButton("Load Model")
        self.browse_model_btn.clicked.connect(self.browse_model)
        self.load_model_btn.clicked.connect(self.load_model)
        model_btn_layout.addWidget(self.browse_model_btn)
        model_btn_layout.addWidget(self.load_model_btn)
        model_layout.addLayout(model_btn_layout)
        
        self.model_status_label = QLabel("Status: No model loaded")
        self.model_status_label.setStyleSheet("color: red;")
        model_layout.addWidget(self.model_status_label)
        
        model_group.setLayout(model_layout)
        layout.addWidget(model_group)
        
        settings_group = QGroupBox("Detection Settings")
        settings_layout = QVBoxLayout()
        
        conf_layout = QHBoxLayout()
        conf_layout.addWidget(QLabel("Confidence Threshold:"))
        self.confidence_slider = QSlider(Qt.Horizontal)
        self.confidence_slider.setRange(10, 100)
        self.confidence_slider.setValue(50)
        self.confidence_slider.valueChanged.connect(self.update_confidence)
        conf_layout.addWidget(self.confidence_slider)
        
        self.confidence_label = QLabel("0.50")
        self.confidence_label.setMinimumWidth(40)
        conf_layout.addWidget(self.confidence_label)
        settings_layout.addLayout(conf_layout)
        
        settings_group.setLayout(settings_layout)
        layout.addWidget(settings_group)
        
        image_group = QGroupBox("Image Detection")
        image_layout = QVBoxLayout()
        
        self.select_image_btn = QPushButton("Select Image")
        self.detect_image_btn = QPushButton("Detect Objects")
        self.save_image_btn = QPushButton("Save Result")
        
        self.select_image_btn.clicked.connect(self.select_image)
        self.detect_image_btn.clicked.connect(self.detect_image)
        self.save_image_btn.clicked.connect(self.save_image)
        
        image_layout.addWidget(self.select_image_btn)
        image_layout.addWidget(self.detect_image_btn)
        image_layout.addWidget(self.save_image_btn)
        
        image_group.setLayout(image_layout)
        layout.addWidget(image_group)
        
        video_group = QGroupBox("Video Detection")
        video_layout = QVBoxLayout()
        
        self.select_video_btn = QPushButton("Select Video File")
        self.start_video_btn = QPushButton("Start Video Detection")
        self.stop_detection_btn = QPushButton("Stop Video Detection")
        self.select_video_btn.clicked.connect(self.select_video)
        self.start_video_btn.clicked.connect(self.start_video_detection)
        self.stop_detection_btn.clicked.connect(self.stop_detection)
        
        video_layout.addWidget(self.select_video_btn)
        video_layout.addWidget(self.start_video_btn)
        video_layout.addWidget(self.stop_detection_btn)
        
        video_group.setLayout(video_layout)
        layout.addWidget(video_group)
        
        live_group = QGroupBox("Live Detection")
        live_layout = QVBoxLayout()
        
        self.save_video_checkbox = QCheckBox("Save video output")
        live_layout.addWidget(self.save_video_checkbox)
        
        self.start_live_btn = QPushButton("Start Live Detection")
        self.stop_detection_btn = QPushButton("Stop Detection")
        
        self.start_live_btn.clicked.connect(self.start_live_detection)
        self.stop_detection_btn.clicked.connect(self.stop_detection)
        
        live_layout.addWidget(self.start_live_btn)
        live_layout.addWidget(self.stop_detection_btn)
        
        live_group.setLayout(live_layout)
        layout.addWidget(live_group)
        
        stats_group = QGroupBox("Detection Statistics")
        stats_layout = QVBoxLayout()
        
        self.stats_label = QLabel("Ready...")
        self.stats_label.setWordWrap(True)
        stats_layout.addWidget(self.stats_label)
        
        stats_group.setLayout(stats_layout)
        layout.addWidget(stats_group)
        
        layout.addStretch()
        
    def setup_display(self, layout):
        display_group = QGroupBox("Detection Display")
        display_layout = QVBoxLayout()
        
        self.image_label = QLabel()
        self.image_label.setAlignment(Qt.AlignCenter)
        self.image_label.setMinimumSize(600, 400)
        self.image_label.setStyleSheet("border: 1px solid gray; background-color: white;")
        self.image_label.setText("Select an image or start detection to see results")
        
        display_layout.addWidget(self.image_label)
        display_group.setLayout(display_layout)
        layout.addWidget(display_group)
        
    def browse_model(self):
        file_path, _ = QFileDialog.getOpenFileName(
            self, "Select YOLO Model", "", "PyTorch files (*.pt);;All files (*)"
        )
        if file_path:
            self.model_path_label.setText(f"Selected: {os.path.basename(file_path)}")
            self.selected_model_path = file_path
            
    def load_model(self):
        if not hasattr(self, 'selected_model_path'):
            QMessageBox.warning(self, "Warning", "Please select a model file first")
            return
            
        try:
            self.model = YOLO(self.selected_model_path)
            self.video_thread.set_model(self.model)
            self.model_status_label.setText("Status: Model loaded successfully")
            self.model_status_label.setStyleSheet("color: green;")
            QMessageBox.information(self, "Success", "Model loaded successfully!")
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Failed to load model: {str(e)}")
            self.model_status_label.setText("Status: Failed to load model")
            self.model_status_label.setStyleSheet("color: red;")
            
    def update_confidence(self, value):
        confidence = value / 100.0
        self.confidence_label.setText(f"{confidence:.2f}")
        self.video_thread.set_confidence(confidence)
        
    def select_image(self):
        if not self.model:
            QMessageBox.warning(self, "Warning", "Please load a model first")
            return
            
        file_path, _ = QFileDialog.getOpenFileName(
            self, "Select Image", "", "Image files (*.jpg *.jpeg *.png *.bmp *.tiff);;All files (*)"
        )
        
        if file_path:
            try:
                self.current_image = cv2.imread(file_path)
                if self.current_image is None:
                    raise Exception("Could not load image")
                self.display_image(self.current_image)
                self.stats_label.setText(f"Image loaded: {os.path.basename(file_path)}")
            except Exception as e:
                QMessageBox.critical(self, "Error", f"Failed to load image: {str(e)}")
                
    def detect_image(self):
        if not self.model:
            QMessageBox.warning(self, "Warning", "Please load a model first")
            return
            
        if self.current_image is None:
            QMessageBox.warning(self, "Warning", "Please select an image first")
            return
            
        try:
            confidence = self.confidence_slider.value() / 100.0
            results = self.model(self.current_image, conf=confidence)
            
            annotated_image = self.video_thread.draw_detections(self.current_image.copy(), results[0])
            
            self.current_image = annotated_image
            self.display_image(annotated_image)
            
            num_detections = len(results[0].boxes) if results[0].boxes is not None else 0
            self.stats_label.setText(f"Detection complete: {num_detections} objects found")
            QMessageBox.information(self, "Detection Complete", f"Found {num_detections} objects")
            
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Detection failed: {str(e)}")
            
    def save_image(self):
        if self.current_image is None:
            QMessageBox.warning(self, "Warning", "No image to save")
            return
            
        file_path, _ = QFileDialog.getSaveFileName(
            self, "Save Image", "", "JPEG files (*.jpg);;PNG files (*.png);;All files (*)"
        )
        
        if file_path:
            try:
                cv2.imwrite(file_path, self.current_image)
                QMessageBox.information(self, "Success", "Image saved successfully!")
            except Exception as e:
                QMessageBox.critical(self, "Error", f"Failed to save image: {str(e)}")
                
    def select_video(self):
        if not self.model:
            QMessageBox.warning(self, "Warning", "Please load a model first")
            return
            
        file_path, _ = QFileDialog.getOpenFileName(
            self, "Select Video", "", "Video files (*.mp4 *.avi *.mov *.mkv);;All files (*)"
        )
        
        if file_path:
            self.selected_video_path = file_path
            self.stats_label.setText(f"Video selected: {os.path.basename(file_path)}")
            
    def start_video_detection(self):
        if not hasattr(self, 'selected_video_path'):
            QMessageBox.warning(self, "Warning", "Please select a video file first")
            return
            
        if self.video_thread.isRunning():
            QMessageBox.warning(self, "Warning", "Detection is already running")
            return
            
        self.video_thread.set_source(self.selected_video_path, save_video=True)
        self.video_thread.start()
        self.stats_label.setText("Processing video... Please wait.")
        
    def start_live_detection(self):
        if not self.model:
            QMessageBox.warning(self, "Warning", "Please load a model first")
            return
            
        if self.video_thread.isRunning():
            QMessageBox.warning(self, "Warning", "Detection is already running")
            return
            
        save_video = self.save_video_checkbox.isChecked()
        self.video_thread.set_source(0, save_video=save_video)  #--> We can change if we use different cameras 0 for default camera
        self.video_thread.start()
        self.stats_label.setText("Live detection started...")
        
    def stop_detection(self):
        if self.video_thread.isRunning():
            self.video_thread.stop()
            self.video_thread.wait() 
            self.stats_label.setText("Detection stopped")
            
    def update_display(self, cv_image):
        self.display_image(cv_image)
        
    def update_stats(self, stats_text):
        self.stats_label.setText(stats_text)
        
    def on_detection_finished(self):
        self.stats_label.setText("Detection finished")
        
    def display_image(self, cv_image):
        try:
            rgb_image = cv2.cvtColor(cv_image, cv2.COLOR_BGR2RGB)
            h, w, ch = rgb_image.shape
            bytes_per_line = ch * w
            
            qt_image = QImage(rgb_image.data, w, h, bytes_per_line, QImage.Format_RGB888)
            
            label_size = self.image_label.size()
            scaled_pixmap = QPixmap.fromImage(qt_image).scaled(
                label_size, Qt.KeepAspectRatio, Qt.SmoothTransformation
            )
            
            self.image_label.setPixmap(scaled_pixmap)
            
        except Exception as e:
            print(f"Display error: {e}")
            
    def closeEvent(self, event):
        if self.video_thread.isRunning():
            self.video_thread.stop()
            self.video_thread.wait()
        event.accept()