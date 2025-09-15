import cv2
import numpy as np
from PyQt5.QtCore import Qt, QThread, pyqtSignal, QMutex
from PyQt5.QtGui import QPixmap, QImage, QFont
from datetime import datetime


class VideoThread(QThread):
    frame_ready = pyqtSignal(np.ndarray)
    detection_stats = pyqtSignal(str)
    finished = pyqtSignal()
    
    def __init__(self):
        super().__init__()
        self.model = None
        self.confidence = 0.5
        self.running = False
        self.source = None
        self.save_video = False
        self.video_writer = None
        self.mutex = QMutex()
        
    def set_model(self, model):
        self.model = model
        
    def set_confidence(self, confidence):
        self.confidence = confidence
        
    def set_source(self, source, save_video=False):
        self.source = source
        self.save_video = save_video
        
    def stop(self):
        self.mutex.lock()
        self.running = False
        self.mutex.unlock()
        
    def run(self):
        if not self.model or self.source is None:
            return
            
        self.running = True
        cap = cv2.VideoCapture(self.source)
        
        if not cap.isOpened():
            self.detection_stats.emit("Error: Could not open video source")
            return
            
        if self.save_video and isinstance(self.source, str):
            fps = int(cap.get(cv2.CAP_PROP_FPS)) or 30
            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_path = f"detection_output_{timestamp}.mp4"
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            self.video_writer = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
            
        frame_count = 0
        detection_count = 0
        
        while self.running and cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
                
            frame_count += 1
            
            try:
                results = self.model(frame, conf=self.confidence, verbose=False)
                
                if results[0].boxes is not None:
                    detection_count += len(results[0].boxes)
                
                annotated_frame = self.draw_detections(frame.copy(), results[0])
                
                if self.save_video and self.video_writer:
                    self.video_writer.write(annotated_frame)
                
                self.frame_ready.emit(annotated_frame)
                
                if frame_count % 30 == 0: 
                    stats = f"Frames: {frame_count} | Detections: {detection_count} | FPS: ~{30/1:.1f}"
                    self.detection_stats.emit(stats)
                    
            except Exception as e:
                self.detection_stats.emit(f"Detection error: {str(e)}")
                
            if self.source == 0:
                self.msleep(33)  
                
        cap.release()
        if self.video_writer:
            self.video_writer.release()
            self.detection_stats.emit(f"Video saved! Total frames: {frame_count}, Total detections: {detection_count}")
            
        self.running = False
        self.finished.emit()
        
    def draw_detections(self, image, results):
        if results.boxes is None:
            return image
            
        for box in results.boxes:
            x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
            conf = box.conf[0].cpu().item()
            cls = int(box.cls[0].cpu().item())

            class_name = results.names[cls]
            
            if conf >= 0.8:
                color = (0, 255, 0)  
            elif conf >= 0.6:
                color = (0, 165, 255)  
            else:
                color = (0, 0, 255)  
            
            cv2.rectangle(image, (int(x1), int(y1)), (int(x2), int(y2)), color, 2)
            
            label = f'{class_name}: {conf:.2f}'
            label_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)[0]
            cv2.rectangle(image, (int(x1), int(y1) - label_size[1] - 10), 
                         (int(x1) + label_size[0], int(y1)), color, -1)
            
            cv2.putText(image, label, (int(x1), int(y1) - 5), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
                       
        return image