import cv2
import threading
import queue
import time
import numpy as np
from datetime import datetime
import os
import json
import pickle
import hashlib
import sys
import re
import psutil
import requests
import zipfile
import torch
import torch.nn as nn
from torchvision import transforms
from ultralytics import YOLO
import torch.ao.quantization.quantize_fx as quantize_fx
from torch.ao.quantization import QConfigMapping
import tkinter as tk
from tkinter import ttk, filedialog, messagebox, simpledialog
from PIL import Image, ImageTk
from collections import deque, Counter
import traceback
import io
import random

original_stdout = sys.stdout
original_stderr = sys.stderr

def generate_stream_id(existing_ids):
    """Генерация уникального ID для потока"""
    while True:
        stream_id = f"stream_{random.randint(10000, 99999)}"
        if stream_id not in existing_ids:
            return stream_id

class ANPRConfig:
    YOLO_MODEL_PATH: str = 'models/anpr/yolo_model/best.pt'
    OCR_MODEL_PATH: str = 'models/ocr_crnn/quant/crnn_ocr_model_int8_fx.pth'
    OCR_IMG_HEIGHT: int = 32
    OCR_IMG_WIDTH: int = 128
    OCR_ALPHABET: str = '0123456789ABCEHKMOPTXY'
    DETECTION_CONFIDENCE_THRESHOLD: float = 0.5
    DEVICE: torch.device = torch.device("cpu")

class ANPRCRNN(nn.Module):
    def __init__(self, num_classes):
        super(ANPRCRNN, self).__init__()
        self.cnn = nn.Sequential(
            nn.Conv2d(1, 64, kernel_size=3, padding=1),
            nn.ReLU(True),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(True),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.ReLU(True),
            nn.MaxPool2d((2, 1), (2, 1)),
            nn.Conv2d(256, 512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(True),
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.ReLU(True),
            nn.MaxPool2d((2, 1), (2, 1))
        )
        self.rnn = nn.LSTM(512 * 2, 256, bidirectional=True, num_layers=2, batch_first=True)
        self.classifier = nn.Linear(512, num_classes)

    def forward(self, x):
        x = self.cnn(x)
        batch, channels, height, width = x.size()
        x = x.reshape(batch, channels * height, width)
        x = x.permute(0, 2, 1)
        x, _ = self.rnn(x)
        x = self.classifier(x)
        x = x.permute(1, 0, 2)
        x = nn.functional.log_softmax(x, dim=2)
        return x

class ANPRYOLODetector:
    def __init__(self, model_path: str, device: torch.device):
        try:
            self.device = device
            self.model = YOLO(model_path)
            if hasattr(torch, 'cuda') and torch.cuda.is_available() and device.type == 'cuda':
                self.model.to(device)
            else:
                self.model.to('cpu')
                self.device = torch.device('cpu')
            print("✅ ANPR Детектор YOLO успешно загружен.")
        except Exception as e:
            print(f"❌ Ошибка загрузки ANPR детектора: {e}")
            self.model = None

    def detect(self, frame: np.ndarray):
        if self.model is None:
            return []
        try:
            device_str = 'cpu'
            if hasattr(self.device, 'type') and self.device.type == 'cuda' and torch.cuda.is_available():
                device_str = '0'
            detections = self.model.predict(frame, verbose=False, device=device_str)
            results = []
            if detections and len(detections) > 0 and hasattr(detections[0], 'boxes'):
                for det in detections[0].boxes.data:
                    if len(det) >= 6:
                        x1, y1, x2, y2, conf, _ = det.cpu().numpy()
                        if conf >= ANPRConfig.DETECTION_CONFIDENCE_THRESHOLD:
                            w = int(x2 - x1)
                            h = int(y2 - y1)
                            x = int(x1)
                            y = int(y1)
                            results.append({
                                "bbox": [x, y, w, h],
                                "bbox_xyxy": [x, y, x + w, y + h],
                                "confidence": float(conf),
                                "class": "license_plate",
                                "class_name": "license_plate",
                                "source": "anpr"
                            })
            return results
        except Exception as e:
            print(f"Ошибка детекции ANPR: {e}")
            return []

class ANPRCRNNRecognizer:
    def __init__(self, model_path: str, device: torch.device):
        self.device = torch.device('cpu')
        self.transform = transforms.Compose([
            transforms.ToPILImage(), transforms.Grayscale(),
            transforms.Resize((ANPRConfig.OCR_IMG_HEIGHT, ANPRConfig.OCR_IMG_WIDTH)),
            transforms.ToTensor(), transforms.Normalize(mean=[0.5], std=[0.5])
        ])
        self.int_to_char = {i + 1: char for i, char in enumerate(ANPRConfig.OCR_ALPHABET)}
        self.int_to_char[0] = ''
        num_classes = len(ANPRConfig.OCR_ALPHABET) + 1
        try:
            model_to_load = ANPRCRNN(num_classes).eval()
            qconfig_mapping = QConfigMapping().set_global(torch.ao.quantization.get_default_qconfig('fbgemm'))
            example_inputs = (torch.randn(1, 1, ANPRConfig.OCR_IMG_HEIGHT, ANPRConfig.OCR_IMG_WIDTH),)
            model_prepared = quantize_fx.prepare_fx(model_to_load, qconfig_mapping, example_inputs)
            model_quantized = quantize_fx.convert_fx(model_prepared)
            if os.path.exists(model_path):
                state_dict = torch.load(model_path, map_location='cpu')
                if isinstance(state_dict, dict) and 'state_dict' in state_dict:
                    state_dict = state_dict['state_dict']
                model_quantized.load_state_dict(state_dict, strict=False)
                self.model = model_quantized
                print("✅ ANPR Распознаватель OCR успешно загружен.")
            else:
                print(f"❌ Файл модели OCR не найден: {model_path}")
                self.model = None
        except Exception as e:
            print(f"❌ Ошибка загрузки OCR модели: {e}")
            traceback.print_exc()
            self.model = None

    @torch.no_grad()
    def recognize(self, plate_image: np.ndarray) -> str:
        if self.model is None or plate_image.size == 0:
            return ""
        try:
            preprocessed_plate = self.transform(plate_image).unsqueeze(0).to(self.device)
            preds = self.model(preprocessed_plate)
            return self._decode(preds)
        except Exception as e:
            print(f"Ошибка распознавания ANPR: {e}")
            traceback.print_exc()
            return ""

    def _decode(self, preds: torch.Tensor) -> str:
        try:
            preds = preds.permute(1, 0, 2).argmax(dim=2)[0]
            decoded_seq = []
            last_char_idx = 0
            for char_idx in preds:
                char_idx = char_idx.item()
                if char_idx != 0 and char_idx != last_char_idx:
                    decoded_seq.append(self.int_to_char.get(char_idx, ''))
                last_char_idx = char_idx
            return "".join(decoded_seq)
        except:
            return ""

class ANPRPipeline:
    def __init__(self, config, model_manager):
        self.config = config
        self.model_manager = model_manager
        self.enabled = config.get('anpr', {}).get('enabled', False)
        self.detector = None
        self.recognizer = None
        self.track_history = {}
        self.TRACK_BUFFER_SIZE = 15
        if self.enabled:
            self.init_anpr()

    def init_anpr(self):
        try:
            device = torch.device("cpu")
            if hasattr(self.model_manager, 'torch_device'):
                try:
                    device = self.model_manager.torch_device
                except:
                    device = torch.device("cpu")
            use_separate_yolo = self.config.get('anpr', {}).get('use_separate_yolo', True)
            if use_separate_yolo:
                detector_path = self.config.get('anpr', {}).get('yolo_model_path', ANPRConfig.YOLO_MODEL_PATH)
                if os.path.exists(detector_path):
                    self.detector = ANPRYOLODetector(detector_path, device)
                    print(f"✅ ANPR детектор загружен: {detector_path}")
                else:
                    print(f"❌ Файл детектора ANPR не найден: {detector_path}")
            recognizer_path = self.config.get('anpr', {}).get('ocr_model_path', ANPRConfig.OCR_MODEL_PATH)
            if os.path.exists(recognizer_path):
                self.recognizer = ANPRCRNNRecognizer(recognizer_path, device)
                print(f"✅ ANPR OCR загружен: {recognizer_path}")
            else:
                print(f"❌ Файл OCR модели ANPR не найден: {recognizer_path}")
            print(f"✅ ANPR система инициализирована: детектор={self.detector is not None and self.detector.model is not None}, распознаватель={self.recognizer is not None and self.recognizer.model is not None}")
        except Exception as e:
            print(f"❌ Ошибка инициализации ANPR: {e}")
            traceback.print_exc()

    def _order_points(self, pts: np.ndarray) -> np.ndarray:
        pts = pts.reshape(4, 2)
        rect = np.zeros((4, 2), dtype="float32")
        s = pts.sum(axis=1)
        rect[0] = pts[np.argmin(s)]
        rect[2] = pts[np.argmax(s)]
        diff = np.diff(pts, axis=1)
        rect[1] = pts[np.argmin(diff)]
        rect[3] = pts[np.argmax(diff)]
        return rect

    def _four_point_transform(self, image: np.ndarray, pts: np.ndarray) -> np.ndarray:
        rect = self._order_points(pts)
        (tl, tr, br, bl) = rect
        widthA = np.sqrt(((br[0] - bl[0]) ** 2) + ((br[1] - bl[1]) ** 2))
        widthB = np.sqrt(((tr[0] - tl[0]) ** 2) + ((tr[1] - tl[1]) ** 2))
        maxWidth = max(int(widthA), int(widthB))
        heightA = np.sqrt(((tr[0] - br[0]) ** 2) + ((tr[1] - br[1]) ** 2))
        heightB = np.sqrt(((tl[0] - bl[0]) ** 2) + ((tl[1] - bl[1]) ** 2))
        maxHeight = max(int(heightA), int(heightB))
        if maxWidth <= 0 or maxHeight <= 0:
            return image
        dst = np.array([
            [0, 0],
            [maxWidth - 1, 0],
            [maxWidth - 1, maxHeight - 1],
            [0, maxHeight - 1]
        ], dtype="float32")
        M = cv2.getPerspectiveTransform(rect, dst)
        return cv2.warpPerspective(image, M, (maxWidth, maxHeight))

    def _detect_skew_angle(self, gray_image: np.ndarray) -> float:
        _, binary = cv2.threshold(gray_image, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
        contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            return 0.0
        largest_contour = max(contours, key=cv2.contourArea)
        rect = cv2.minAreaRect(largest_contour)
        angle = rect[2]
        if angle < -45:
            angle = 90 + angle
        return angle

    def _rotate_image(self, image: np.ndarray, angle: float) -> np.ndarray:
        if angle == 0:
            return image
        (h, w) = image.shape[:2]
        center = (w // 2, h // 2)
        M = cv2.getRotationMatrix2D(center, angle, 1.0)
        rotated = cv2.warpAffine(image, M, (w, h),
                                flags=cv2.INTER_CUBIC,
                                borderMode=cv2.BORDER_REPLICATE)
        return rotated

    def _find_plate_corners(self, gray_image: np.ndarray):
        binary = cv2.adaptiveThreshold(gray_image, 255,
                                      cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                      cv2.THRESH_BINARY_INV, 11, 2)
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
        binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)
        contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            return None
        plate_contours = []
        for contour in contours:
            area = cv2.contourArea(contour)
            if area < 100:
                continue
            peri = cv2.arcLength(contour, True)
            approx = cv2.approxPolyDP(contour, 0.02 * peri, True)
            if len(approx) == 4:
                plate_contours.append(approx)
            else:
                rect = cv2.minAreaRect(contour)
                box = cv2.boxPoints(rect)
                box = np.int0(box)
                plate_contours.append(box.reshape(-1, 1, 2))
        if not plate_contours:
            return None
        plate_contour = max(plate_contours, key=cv2.contourArea)
        return plate_contour.reshape(4, 2)

    def _preprocess_plate(self, plate_image: np.ndarray) -> np.ndarray:
        if plate_image.size == 0:
            return plate_image
        try:
            gray = cv2.cvtColor(plate_image, cv2.COLOR_BGR2GRAY)
            gray = cv2.equalizeHist(gray)
            gray = cv2.GaussianBlur(gray, (3, 3), 0)
            skew_angle = self._detect_skew_angle(gray)
            if abs(skew_angle) > 1.0:
                rotated = self._rotate_image(plate_image, skew_angle)
                gray_rotated = cv2.cvtColor(rotated, cv2.COLOR_BGR2GRAY)
                gray_rotated = cv2.equalizeHist(gray_rotated)
                gray_rotated = cv2.GaussianBlur(gray_rotated, (3, 3), 0)
            else:
                rotated = plate_image
                gray_rotated = gray
            corners = self._find_plate_corners(gray_rotated)
            if corners is not None:
                try:
                    warped = self._four_point_transform(rotated, corners)
                    h, w = warped.shape[:2]
                    aspect_ratio = w / h
                    if 1.5 < aspect_ratio < 5.0:
                        return warped
                    else:
                        return rotated
                except Exception as e:
                    print(f"Ошибка при перспективном преобразовании: {e}")
                    return rotated
            else:
                edges = cv2.Canny(gray_rotated, 50, 150)
                lines = cv2.HoughLinesP(edges, 1, np.pi/180, 50, minLineLength=30, maxLineGap=10)
                if lines is not None:
                    all_points = []
                    for line in lines:
                        x1, y1, x2, y2 = line[0]
                        all_points.append([x1, y1])
                        all_points.append([x2, y2])
                    if all_points:
                        points = np.array(all_points)
                        hull = cv2.convexHull(points)
                        peri = cv2.arcLength(hull, True)
                        approx = cv2.approxPolyDP(hull, 0.02 * peri, True)
                        if len(approx) == 4:
                            try:
                                warped = self._four_point_transform(rotated, approx.reshape(4, 2))
                                return warped
                            except:
                                pass
            return rotated
        except Exception as e:
            print(f"Ошибка в предобработке номера: {e}")
            traceback.print_exc()
            return plate_image

    def _stabilize_text(self, track_id: int, new_text: str) -> str:
        if track_id not in self.track_history:
            self.track_history[track_id] = []
        self.track_history[track_id].append(new_text)
        if len(self.track_history[track_id]) > self.TRACK_BUFFER_SIZE:
            self.track_history[track_id].pop(0)
        counts = Counter(self.track_history[track_id])
        if not counts:
            return ""
        best_text, best_count = counts.most_common(1)[0]
        return best_text if best_count >= 1 else ""

    def detect_plates(self, frame: np.ndarray):
        if not self.enabled or not self.detector or self.detector.model is None:
            return []
        try:
            return self.detector.detect(frame)
        except Exception as e:
            print(f"Ошибка детекции ANPR: {e}")
            return []

    def recognize_plate(self, plate_image: np.ndarray, track_id: int = None) -> str:
        if not self.enabled or not self.recognizer or self.recognizer.model is None or plate_image.size == 0:
            return ""
        try:
            processed_plate = self._preprocess_plate(plate_image)
            if processed_plate.size == 0:
                return ""
            current_text = self.recognizer.recognize(processed_plate)
            if track_id is not None and current_text:
                return self._stabilize_text(track_id, current_text)
            return current_text
        except Exception as e:
            print(f"Ошибка распознавания ANPR: {e}")
            traceback.print_exc()
            return ""

    def process_detections(self, frame: np.ndarray, detections):
        if not self.enabled:
            return detections
        results = []
        for detection in detections:
            if 'bbox' in detection:
                bbox = detection['bbox']
                if len(bbox) == 4:
                    x, y, w, h = bbox
                elif 'bbox_xyxy' in detection and len(detection['bbox_xyxy']) == 4:
                    x1, y1, x2, y2 = detection['bbox_xyxy']
                    x, y, w, h = x1, y1, x2 - x1, y2 - y1
                else:
                    results.append(detection)
                    continue
                x, y, w, h = int(x), int(y), int(w), int(h)
                if w < 20 or h < 8:
                    results.append(detection)
                    continue
                height, width = frame.shape[:2]
                x = max(0, min(x, width - 1))
                y = max(0, min(y, height - 1))
                w = max(1, min(w, width - x))
                h = max(1, min(h, height - y))
                roi = frame[y:y+h, x:x+w]
                if roi.size > 0:
                    track_id = detection.get('track_id')
                    plate_text = self.recognize_plate(roi, track_id)
                    if plate_text:
                        detection['plate_text'] = plate_text
                        detection['class_name'] = f"Номер: {plate_text}"
                        detection['anpr_confidence'] = detection.get('confidence', 0.5)
                        detection['recognized'] = True
                        detection['recognized_at'] = datetime.now().isoformat()
                        detection['bbox'] = [x, y, w, h]
                        print(f"✅ Распознан номер: {plate_text} (размер: {w}x{h}, уверенность: {detection['anpr_confidence']:.2f})")
                    else:
                        detection['plate_text'] = ""
                        detection['recognized'] = False
                        print(f"❌ Номер не распознан (размер ROI: {roi.shape[:2]})")
            results.append(detection)
        return results

try:
    import torch
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False

try:
    from ultralytics import YOLO
    HAS_YOLO = True
except ImportError:
    HAS_YOLO = False

try:
    from onvif import ONVIFCamera
    HAS_ONVIF = True
except ImportError:
    HAS_ONVIF = False

class SuppressOutput:
    def __enter__(self):
        self.original_stdout = sys.stdout
        self.original_stderr = sys.stderr
        sys.stdout = io.StringIO()
        sys.stderr = io.StringIO()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        sys.stdout = self.original_stdout
        sys.stderr = self.original_stderr

class DeviceDetector:
    @staticmethod
    def detect_devices():
        devices = {
            'cpu': {'available': True, 'name': 'CPU', 'type': 'cpu', 'priority': 0},
            'amd_gpu': {'available': False, 'name': 'AMD GPU', 'type': 'gpu', 'priority': 1},
            'intel_gpu': {'available': False, 'name': 'Intel GPU', 'type': 'gpu', 'priority': 2},
            'nvidia_gpu': {'available': False, 'name': 'NVIDIA GPU', 'type': 'gpu', 'priority': 3}
        }
        if HAS_TORCH:
            try:
                if torch.cuda.is_available():
                    devices['nvidia_gpu']['available'] = True
                    devices['nvidia_gpu']['name'] = f'NVIDIA GPU ({torch.cuda.get_device_name(0)})'
                    devices['nvidia_gpu']['device_count'] = torch.cuda.device_count()
            except:
                devices['nvidia_gpu']['available'] = False
            try:
                if hasattr(torch, 'version') and hasattr(torch.version, 'hip'):
                    devices['amd_gpu']['available'] = True
                    devices['amd_gpu']['name'] = 'AMD GPU (ROCm)'
            except:
                pass
            try:
                if hasattr(torch, 'xpu') and torch.xpu.is_available():
                    devices['intel_gpu']['available'] = True
                    devices['intel_gpu']['name'] = 'Intel GPU (XPU)'
            except:
                pass
        try:
            cpu_info = f"CPU Cores: {psutil.cpu_count(logical=False)}"
            devices['cpu']['name'] = f'CPU ({cpu_info})'
        except:
            pass
        return devices

    @staticmethod
    def get_optimal_device(preference='auto'):
        devices = DeviceDetector.detect_devices()
        if preference == 'cpu':
            return 'cpu', devices['cpu']
        device_order = ['nvidia_gpu', 'amd_gpu', 'intel_gpu', 'cpu']
        if preference == 'amd':
            device_order.insert(0, 'amd_gpu')
        elif preference == 'intel':
            device_order.insert(0, 'intel_gpu')
        elif preference == 'nvidia':
            device_order.insert(0, 'nvidia_gpu')
        for device_key in device_order:
            if devices[device_key]['available']:
                return device_key, devices[device_key]
        return 'cpu', devices['cpu']

    @staticmethod
    def setup_torch_device(device_key):
        try:
            import torch
            if device_key == 'cpu':
                return torch.device('cpu'), 'cpu'
            elif device_key == 'nvidia_gpu' and torch.cuda.is_available():
                return torch.device('cuda:0'), 'cuda:0'
            elif device_key == 'amd_gpu':
                try:
                    if hasattr(torch.version, 'hip'):
                        return torch.device('cuda:0'), 'cuda:0'
                    else:
                        return torch.device('cpu'), 'cpu'
                except:
                    return torch.device('cpu'), 'cpu'
            elif device_key == 'intel_gpu':
                try:
                    if hasattr(torch, 'xpu') and torch.xpu.is_available():
                        return torch.device('xpu:0'), 'xpu:0'
                except:
                    pass
            return torch.device('cpu'), 'cpu'
        except ImportError:
            return None, 'cpu'

class ModelManager:
    def __init__(self, progress_callback=None, device_preference='auto', config=None):
        self.models_dir = "models"
        self.progress_callback = progress_callback
        self.device_preference = device_preference
        self.config = config or {}
        self.devices_info = DeviceDetector.detect_devices()
        self.device_key, self.device_info = DeviceDetector.get_optimal_device(device_preference)
        self.torch_device, self.torch_device_str = DeviceDetector.setup_torch_device(self.device_key)
        self.available_yolo_models = {
            'yolo26n': {'name': 'yolo26n.pt', 'display': 'YOLOv26 Nano (n)', 'size': 'small', 'speed': 'fastest'},
            'yolo26s': {'name': 'yolo26s.pt', 'display': 'YOLOv26 Small (s)', 'size': 'small', 'speed': 'fast'},
            'yolo26m': {'name': 'yolo26m.pt', 'display': 'YOLOv26 Medium (m)', 'size': 'medium', 'speed': 'medium'},
            'yolo26l': {'name': 'yolo26l.pt', 'display': 'YOLOv26 Large (l)', 'size': 'large', 'speed': 'slow'},
            'yolo26x': {'name': 'yolo26x.pt', 'display': 'YOLOv26 XLarge (x)', 'size': 'xlarge', 'speed': 'slowest'}
        }
        self.models = {
            'yolo26n': {
                'type': 'object_detection',
                'model_name': 'yolo26n.pt',
                'required': True,
                'ultralytics_model': True,
                'yolo_name': 'yolo26n.pt',
                'local_path': 'models/yolo26n.pt'
            },
            'yolo26s': {
                'type': 'object_detection',
                'model_name': 'yolo26s.pt',
                'required': False,
                'ultralytics_model': True,
                'yolo_name': 'yolo26s.pt',
                'local_path': 'models/yolo26s.pt'
            },
            'yolo26m': {
                'type': 'object_detection',
                'model_name': 'yolo26m.pt',
                'required': False,
                'ultralytics_model': True,
                'yolo_name': 'yolo26m.pt',
                'local_path': 'models/yolo26m.pt'
            },
            'yolo26l': {
                'type': 'object_detection',
                'model_name': 'yolo26l.pt',
                'required': False,
                'ultralytics_model': True,
                'yolo_name': 'yolo26l.pt',
                'local_path': 'models/yolo26l.pt'
            },
            'yolo26x': {
                'type': 'object_detection',
                'model_name': 'yolo26x.pt',
                'required': False,
                'ultralytics_model': True,
                'yolo_name': 'yolo26x.pt',
                'local_path': 'models/yolo26x.pt'
            },
            'anpr_yolo': {
                'type': 'anpr_detection',
                'model_name': 'anpr_yolo.pt',
                'required': False,
                'ultralytics_model': True,
                'yolo_name': 'anpr_yolo.pt',
                'local_path': 'models/anpr/yolo_model/best.pt'
            },
            'anpr_ocr': {
                'type': 'anpr_recognition',
                'model_name': 'anpr_ocr.pth',
                'required': False,
                'local_path': 'models/ocr_crnn/quant/crnn_ocr_model_int8_fx.pth'
            },
            'materials_yolo': {
                'type': 'materials_detection',
                'model_name': 'materials.pt',
                'required': False,
                'ultralytics_model': True,
                'yolo_name': 'materials.pt',
                'local_path': 'models/materials/materials.pt'
            }
        }
        self.create_directories()
        self.yolo_model = None
        self.yolo_model_name = None
        self.materials_yolo_model = None
        self.anpr_pipeline = None
        self.yolo_initialized = False
        if self.progress_callback:
            self.progress_callback(f"Используется устройство: {self.device_info['name']}",
                                 detail=f"Тип: {self.device_key}")

    def create_directories(self):
        os.makedirs(self.models_dir, exist_ok=True)
        os.makedirs('models/cache', exist_ok=True)
        os.makedirs('models/anpr/yolo_model', exist_ok=True)
        os.makedirs('models/ocr_crnn/quant', exist_ok=True)
        os.makedirs('models/materials', exist_ok=True)

    def init_models_after_download(self):
        self.init_yolo()

    def init_yolo(self):
        if not HAS_YOLO:
            if self.progress_callback:
                self.progress_callback("YOLO не установлен",
                                     detail="Установите: pip install ultralytics",
                                     progress=100)
            return False
        try:
            yolo_models_dir = self.config.get('detection', {}).get('yolo_models_dir', 'models')
            selected_model = self.config.get('detection', {}).get('yolo_model', 'yolo26n.pt')
            model_versions = []
            for model_key, model_info in self.available_yolo_models.items():
                if model_info['name'] == selected_model:
                    model_versions.append({'name': model_key, 'file': selected_model})
            for model_key, model_info in self.available_yolo_models.items():
                if model_info['name'] != selected_model:
                    model_versions.append({'name': model_key, 'file': model_info['name']})
            for model_info in model_versions:
                try:
                    model_name = model_info['name']
                    model_file = model_info['file']
                    local_path = os.path.join(yolo_models_dir, model_file)
                    if self.progress_callback:
                        self.progress_callback(f"Загрузка YOLO модели {model_file}...",
                                             detail="Проверка локальной копии",
                                             progress=50)
                    if not os.path.exists(local_path):
                        if self.progress_callback:
                            self.progress_callback(f"Локальная модель {model_file} не найдена",
                                                 detail="Поместите модель в папку models/",
                                                 progress=60)
                        continue
                    with SuppressOutput():
                        self.yolo_model = YOLO(local_path)
                        self.yolo_model_name = model_file
                        if hasattr(self.yolo_model, 'model'):
                            test_img = np.zeros((640, 640, 3), dtype=np.uint8)
                            device = 'cpu'
                            if HAS_TORCH and torch.cuda.is_available():
                                if self.device_key == 'nvidia_gpu':
                                    device = '0'
                                elif self.device_key == 'amd_gpu':
                                    device = '0'
                            if self.progress_callback:
                                self.progress_callback(f"Тестирование YOLO {model_file}...",
                                                     detail=f"На {self.device_info['name']}",
                                                     progress=70)
                            test_results = self.yolo_model(test_img, verbose=False, device=device)
                            if self.progress_callback:
                                self.progress_callback(f"YOLO модель {model_file} загружена успешно",
                                                     detail=f"YOLO v26 на {self.device_info['name']}",
                                                     progress=100)
                            self.yolo_initialized = True
                            return True
                        else:
                            if self.progress_callback:
                                self.progress_callback(f"YOLO {model_file} не загрузился",
                                                     detail="Ошибка инициализации модели",
                                                     progress=90)
                except Exception as e:
                    error_msg = str(e)
                    if self.progress_callback:
                        self.progress_callback(f"Ошибка загрузки YOLO {model_file}",
                                             detail=f"{error_msg[:50]}...",
                                             progress=80)
                    print(f"YOLO {model_file} error: {error_msg}")
                    continue
            if self.progress_callback:
                self.progress_callback("Не удалось загрузить YOLO v26 модель",
                                     detail="Поместите модель в папку models/ (yolo26n.pt, yolo26s.pt или yolo26m.pt)",
                                     progress=100)
            self.yolo_initialized = False
            return False
        except Exception as e:
            error_msg = str(e)
            if self.progress_callback:
                self.progress_callback(f"Критическая ошибка инициализации YOLО",
                                     detail=f"{error_msg[:50]}...",
                                     progress=100)
            print(f"YOLO initialization error: {error_msg}")
            self.yolo_initialized = False
            return False

    def init_materials_yolo(self):
        """Инициализация YOLO модели для детекции материалов"""
        if not HAS_YOLO:
            return False
        
        try:
            materials_model_path = 'models/materials/materials.pt'
            if not os.path.exists(materials_model_path):
                if self.progress_callback:
                    self.progress_callback("Модель для материалов не найдена",
                                         detail=f"Поместите модель materials.pt в {materials_model_path}",
                                         progress=90)
                return False
            
            with SuppressOutput():
                self.materials_yolo_model = YOLO(materials_model_path)
                if hasattr(self.materials_yolo_model, 'model'):
                    # Тестирование модели
                    test_img = np.zeros((640, 640, 3), dtype=np.uint8)
                    device = 'cpu'
                    if HAS_TORCH and torch.cuda.is_available():
                        if self.device_key == 'nvidia_gpu':
                            device = '0'
                        elif self.device_key == 'amd_gpu':
                            device = '0'
                    
                    if self.progress_callback:
                        self.progress_callback("Тестирование модели материалов...",
                                             detail=f"На {self.device_info['name']}",
                                             progress=95)
                    
                    test_results = self.materials_yolo_model(test_img, verbose=False, device=device)
                    
                    if self.progress_callback:
                        self.progress_callback("Модель для материалов загружена успешно",
                                             detail=f"YOLO v26 для материалов",
                                             progress=100)
                    
                    return True
        except Exception as e:
            error_msg = str(e)
            if self.progress_callback:
                self.progress_callback(f"Ошибка загрузки модели материалов",
                                     detail=f"{error_msg[:50]}...",
                                     progress=90)
            print(f"Materials YOLO initialization error: {error_msg}")
            return False
        
        return False

    def init_anpr(self):
        try:
            anpr_yolo_path = 'models/anpr/yolo_model/best.pt'
            anpr_ocr_path = 'models/ocr_crnn/quant/crnn_ocr_model_int8_fx.pth'
            anpr_yolo_exists = os.path.exists(anpr_yolo_path)
            anpr_ocr_exists = os.path.exists(anpr_ocr_path)
            if anpr_yolo_exists and anpr_ocr_exists:
                if self.progress_callback:
                    self.progress_callback("ANPR модели обнаружены",
                                         detail="Готово к инициализации")
                return True
            else:
                if self.progress_callback:
                    self.progress_callback("ANPR модели не найдены",
                                         detail="Для использования ANPR загрузите модели")
                return False
        except Exception as e:
            if self.progress_callback:
                self.progress_callback(f"Ошибка инициализации ANPR",
                                     detail=f"{str(e)[:50]}")
            return False

    def download_all_models(self):
        results = {}
        if self.progress_callback:
            self.progress_callback(f"Инициализация моделей...", progress=80)
        self.init_models_after_download()
        return results

    def get_yolo_model(self):
        if not HAS_YOLO:
            return None
        if self.yolo_model is None or not self.yolo_initialized:
            success = self.init_yolo()
            if not success:
                return None
        return self.yolo_model

    def get_yolo_version(self):
        return self.yolo_model_name

    def get_materials_yolo_model(self):
        """Получение YOLO модели для детекции материалов"""
        if not HAS_YOLO:
            return None
        if self.materials_yolo_model is None:
            success = self.init_materials_yolo()
            if not success:
                return None
        return self.materials_yolo_model

    def get_anpr_pipeline(self):
        if self.anpr_pipeline is None and self.config:
            try:
                self.anpr_pipeline = ANPRPipeline(self.config, self)
                if self.anpr_pipeline and self.anpr_pipeline.enabled:
                    print(f"✅ ANPR Pipeline создан: enabled={self.anpr_pipeline.enabled}")
                    if hasattr(self.anpr_pipeline, 'detector') and self.anpr_pipeline.detector:
                        print(f"   Детектор: {self.anpr_pipeline.detector.model is not None}")
                    if hasattr(self.anpr_pipeline, 'recognizer') and self.anpr_pipeline.recognizer:
                        print(f"   Распознаватель: {self.anpr_pipeline.recognizer.model is not None}")
                else:
                    print(f"❌ ANPR Pipeline не активирован")
            except Exception as e:
                print(f"Ошибка создания ANPR пайплайна: {e}")
                traceback.print_exc()
                return None
        return self.anpr_pipeline

    def get_device_info(self):
        return {
            'device_key': self.device_key,
            'device_info': self.device_info,
            'torch_device': self.torch_device_str,
            'all_devices': self.devices_info,
            'yolo_version': self.yolo_model_name,
            'yolo_initialized': self.yolo_initialized,
            'materials_yolo_available': self.materials_yolo_model is not None
        }

    def is_yolo_available(self):
        return self.yolo_initialized

    def get_available_yolo_models(self):
        available = []
        yolo_models_dir = self.config.get('detection', {}).get('yolo_models_dir', 'models')
        for model_key, model_info in self.available_yolo_models.items():
            local_path = os.path.join(yolo_models_dir, model_info["name"])
            if os.path.exists(local_path):
                model_info_copy = model_info.copy()
                model_info_copy['available'] = True
                model_info_copy['key'] = model_key
                model_info_copy['path'] = local_path
            else:
                model_info_copy = model_info.copy()
                model_info_copy['available'] = False
                model_info_copy['key'] = model_key
                model_info_copy['path'] = local_path
            available.append(model_info_copy)
        return available

class LoadingWindow:
    def __init__(self, root):
        self.root = root
        self.loading_window = tk.Toplevel(root)
        self.loading_window.title("Загрузка AI Видеоанализатора")
        self.loading_window.geometry("500x450")
        self.loading_window.resizable(False, False)
        self.loading_window.attributes('-topmost', True)
        self.center_window()
        bg_color = '#0f172a'
        accent_color = '#3b82f6'
        success_color = '#10b981'
        text_color = '#ffffff'
        self.loading_window.configure(bg=bg_color)
        content_frame = tk.Frame(self.loading_window, bg=bg_color)
        content_frame.pack(fill=tk.BOTH, expand=True, padx=40, pady=40)
        icon_label = tk.Label(content_frame, text="🚀",
                             font=('Segoe UI', 64),
                             bg=bg_color, fg=accent_color)
        icon_label.pack(pady=(0, 20))
        title_label = tk.Label(content_frame, text="AI Видеоанализатор PRO",
                              font=('Segoe UI', 22, 'bold'),
                              bg=bg_color, fg=text_color)
        title_label.pack(pady=(0, 10))
        version_label = tk.Label(content_frame, text="Версия 3.2 с Multi-GPU поддержкой",
                                font=('Segoe UI', 12),
                                bg=bg_color, fg='#94a3b8')
        version_label.pack(pady=(0, 30))
        self.device_frame = tk.Frame(content_frame, bg='#1e293b', relief=tk.RAISED, borderwidth=1)
        self.device_frame.pack(fill=tk.X, pady=(0, 20), padx=10)
        device_title = tk.Label(self.device_frame, text="⚙️ Вычислительное устройство",
                               font=('Segoe UI', 10, 'bold'),
                               bg='#1e293b', fg=text_color)
        device_title.pack(pady=(8, 5))
        self.device_label = tk.Label(self.device_frame, text="Определение устройств...",
                                    font=('Segoe UI', 9),
                                    bg='#1e293b', fg='#94a3b8')
        self.device_label.pack(pady=(0, 8))
        self.progress_var = tk.DoubleVar()
        style = ttk.Style()
        style.theme_use('clam')
        style.configure("Custom.Horizontal.TProgressbar",
                       thickness=30,
                       troughcolor='#334155',
                       background=success_color,
                       lightcolor=success_color,
                       darkcolor=success_color,
                       bordercolor=bg_color)
        self.progress = ttk.Progressbar(content_frame,
                                       style="Custom.Horizontal.TProgressbar",
                                       variable=self.progress_var,
                                       maximum=100,
                                       mode='determinate',
                                       length=400)
        self.progress.pack(pady=(0, 15))
        self.status_label = tk.Label(content_frame, text="Инициализация системы...",
                                    font=('Segoe UI', 12),
                                    bg=bg_color, fg=text_color)
        self.status_label.pack()
        self.detail_label = tk.Label(content_frame, text="",
                                   font=('Segoe UI', 10),
                                   bg=bg_color, fg='#94a3b8')
        self.detail_label.pack(pady=(10, 0))
        self.percent_label = tk.Label(content_frame, text="0%",
                                     font=('Segoe UI', 14, 'bold'),
                                     bg=bg_color, fg=accent_color)
        self.percent_label.pack(pady=(5, 0))
        self.model_label = tk.Label(content_frame, text="",
                                   font=('Segoe UI', 9),
                                   bg=bg_color, fg='#64748b')
        self.model_label.pack(pady=(5, 0))
        copyright_label = tk.Label(content_frame, text="© 2024 AI Video Analytics | Multi-GPU поддержка",
                                  font=('Segoe UI', 9),
                                  bg=bg_color, fg='#64748b')
        copyright_label.pack(side=tk.BOTTOM, pady=(20, 0))
        self.loading_window.lift()
        self.loading_window.focus_force()
        self.loading_thread = None
        self.loading_complete = threading.Event()
        self.root.update()

    def center_window(self):
        self.loading_window.update_idletasks()
        width = self.loading_window.winfo_width()
        height = self.loading_window.winfo_height()
        x = (self.loading_window.winfo_screenwidth() // 2) - (width // 2)
        y = (self.loading_window.winfo_screenheight() // 2) - (height // 2)
        self.loading_window.geometry(f'{width}x{height}+{x}+{y}')

    def start_loading_in_thread(self, loading_function):
        self.loading_thread = threading.Thread(target=loading_function, daemon=True)
        self.loading_thread.start()
        self.check_loading_complete()

    def check_loading_complete(self):
        if self.loading_complete.is_set():
            self.close()
        else:
            self.loading_window.after(100, self.check_loading_complete)

    def update_device_info(self, device_info):
        if self.loading_window.winfo_exists():
            device_text = f"{device_info['name']} ({device_info['type'].upper()})"
            self.device_label.config(text=device_text)
            self.loading_window.update()

    def update_progress(self, value, status_text, detail_text="", model_text=""):
        if self.loading_window.winfo_exists():
            self.progress_var.set(value)
            self.percent_label.config(text=f"{int(value)}%")
            self.status_label.config(text=status_text)
            if detail_text:
                self.detail_label.config(text=detail_text)
            if model_text:
                self.model_label.config(text=model_text)
            self.loading_window.update()

    def close(self):
        if self.loading_window.winfo_exists():
            self.loading_window.destroy()
        if self.root and self.root.winfo_exists():
            self.root.after(0, self.root.focus_force)

def validate_ip(ip_address):
    pattern = r'^(\d{1,3}\.){3}\d{1,3}$'
    if re.match(pattern, ip_address):
        parts = ip_address.split('.')
        if all(0 <= int(part) <= 255 for part in parts):
            return True
    return False

def validate_port(port):
    try:
        port = int(port)
        return 1 <= port <= 65535
    except:
        return False

class Config:
    def __init__(self):
        self.config_file = "config.json"
        self.default_config = {
            "streams": [],
            "detection": {
                "confidence_threshold": 0.5,
                "track_objects": True,
                "yolo_model": "yolo26n.pt",
                "yolo_model_type": "yolo26n",
                "yolo_models_dir": "models",
                "device_preference": "auto",
                "use_gpu_acceleration": True,
                "skip_frames": 1,
                "batch_size": 1,
                "max_queue_size": 1
            },
            "recognition": {
                "faces": True,
                "plates": True,
                "vehicles": True,
                "animals": False,
                "materials": True
            },
            "materials": {
                "enabled": True,
                "materials_list": [
                    "песок",
                    "щебень",
                    "гравий",
                    "глина",
                    "грунт",
                    "асфальт",
                    "бетон"
                ],
                "detection_threshold": 0.5,
                "use_yolo": True,
                "yolo_model_path": "models/materials/materials.pt"
            },
            "database": {
                "faces_dir": "database/faces",
                "plates_dir": "database/plates",
                "vehicles_dir": "database/vehicles",
                "animals_dir": "database/animals",
                "materials_dir": "database/materials"
            },
            "models": {
                "yolo": "yolo26n.pt"
            },
            "save_path": "saved_frames",
            "save_settings": {
                "save_every_n_frame": 5,
                "max_frames_per_event": 50
            },
            "performance": {
                "parallel_processing": True,
                "max_gpu_streams": 1,
                "cpu_workers": 1,
                "batch_size": 1,
                "ui_update_interval": 33,
                "frame_skip_factor": 1
            },
            "anpr": {
                "enabled": False,
                "yolo_model_path": "models/anpr/yolo_model/best.pt",
                "ocr_model_path": "models/ocr_crnn/quant/crnn_ocr_model_int8_fx.pth",
                "detection_confidence_threshold": 0.5,
                "use_separate_yolo": True,
                "recognize_vehicles": True,
                "alphabet": "0123456789ABCEHKMOPTXY",
                "ocr_img_height": 32,
                "ocr_img_width": 128
            }
        }
        self.load_config()

    def load_config(self):
        if os.path.exists(self.config_file):
            try:
                with open(self.config_file, 'r', encoding='utf-8') as f:
                    loaded = json.load(f)
                    self.config = self.deep_update(self.default_config, loaded)
                    streams = self.config.get('streams', [])
                    seen_ids = set()
                    duplicate_found = False
                    for stream in streams:
                        stream_id = stream.get('id')
                        if stream_id in seen_ids:
                            existing_ids = [s['id'] for s in streams]
                            stream['id'] = generate_stream_id(existing_ids)
                            duplicate_found = True
                        seen_ids.add(stream.get('id'))
                    if duplicate_found:
                        print("⚠️ Обнаружены дублирующиеся ID потоков. Исправлено автоматически.")
                        self.save_config()
            except Exception as e:
                print(f"Ошибка загрузки конфигурации: {e}")
                self.config = self.default_config.copy()
        else:
            self.config = self.default_config.copy()
        return self.config

    def save_config(self):
        try:
            with open(self.config_file, 'w', encoding='utf-8') as f:
                json.dump(self.config, f, indent=2, ensure_ascii=False)
        except Exception as e:
            print(f"Ошибка сохранения конфигурации: {e}")

    def deep_update(self, d, u):
        for k, v in u.items():
            if isinstance(v, dict) and k in d and isinstance(d[k], dict):
                self.deep_update(d[k], v)
            else:
                d[k] = v
        return d

    def add_material(self, material_name):
        if 'materials' not in self.config:
            self.config['materials'] = {'enabled': True, 'materials_list': [], 'use_yolo': True}
        if 'materials_list' not in self.config['materials']:
            self.config['materials']['materials_list'] = []
        if material_name not in self.config['materials']['materials_list']:
            self.config['materials']['materials_list'].append(material_name)
            self.save_config()
            return True
        return False

    def remove_material(self, material_name):
        if 'materials' in self.config and 'materials_list' in self.config['materials']:
            if material_name in self.config['materials']['materials_list']:
                self.config['materials']['materials_list'].remove(material_name)
                self.save_config()
                return True
        return False

    def get_materials(self):
        if 'materials' in self.config and 'materials_list' in self.config['materials']:
            return self.config['materials']['materials_list']
        return []

    def is_materials_enabled(self):
        if 'materials' in self.config:
            return self.config['materials'].get('enabled', True)
        return True

    def is_materials_yolo_enabled(self):
        if 'materials' in self.config:
            return self.config['materials'].get('use_yolo', True)
        return True

class ObjectDatabase:
    def __init__(self, config):
        self.config = config
        self.db_file = "database/objects.db"
        self.encodings_file = "database/face_encodings.pkl"
        self.face_encodings = {}
        self.face_metadata = {}
        self.create_directories()
        self.load_database()

    def create_directories(self):
        dirs = [
            self.config['database']['faces_dir'],
            self.config['database']['plates_dir'],
            self.config['database']['vehicles_dir'],
            self.config['database']['animals_dir'],
            self.config['database'].get('materials_dir', 'database/materials'),
            "database/temp",
            "database/backup"
        ]
        for dir_path in dirs:
            os.makedirs(dir_path, exist_ok=True)

    def load_database(self):
        if os.path.exists(self.db_file):
            try:
                with open(self.db_file, 'r', encoding='utf-8') as f:
                    self.database = json.load(f)
            except:
                self.database = {"faces": [], "plates": [], "vehicles": [], "animals": [], "materials": []}
        else:
            self.database = {"faces": [], "plates": [], "vehicles": [], "animals": [], "materials": []}
        self.faces = {}
        self.plates = {}
        self.vehicles = {}
        self.animals = {}
        self.materials = {}
        for face in self.database.get('faces', []):
            self.faces[face['id']] = face
        for plate in self.database.get('plates', []):
            self.plates[plate['id']] = plate
        for vehicle in self.database.get('vehicles', []):
            self.vehicles[vehicle['id']] = vehicle
        for animal in self.database.get('animals', []):
            self.animals[animal['id']] = animal
        for material in self.database.get('materials', []):
            self.materials[material['id']] = material

    def load_face_encodings(self):
        if os.path.exists(self.encodings_file):
            try:
                with open(self.encodings_file, 'rb') as f:
                    data = pickle.load(f)
                    self.face_encodings = {}
                    self.face_metadata = {}
                    for key, value in data.get('encodings', {}).items():
                        self.face_encodings[key] = value
                    for key, value in data.get('metadata', {}).items():
                        self.face_metadata[key] = value
            except:
                self.face_encodings = {}
                self.face_metadata = {}

    def save_face_encodings(self):
        os.makedirs(os.path.dirname(self.encodings_file), exist_ok=True)
        data = {
            'encodings': self.face_encodings,
            'metadata': self.face_metadata
        }
        with open(self.encodings_file, 'wb') as f:
            pickle.dump(data, f)

    def save_database(self):
        self.database['faces'] = list(self.faces.values())
        self.database['plates'] = list(self.plates.values())
        self.database['vehicles'] = list(self.vehicles.values())
        self.database['animals'] = list(self.animals.values())
        self.database['materials'] = list(self.materials.values())
        with open(self.db_file, 'w', encoding='utf-8') as f:
            json.dump(self.database, f, indent=2, ensure_ascii=False)

    def add_face(self, name, image_path, position="", notes=""):
        face_id = hashlib.md5(f"{name}_{datetime.now().timestamp()}".encode()).hexdigest()[:12]
        target_path = os.path.join(self.config['database']['faces_dir'], f"{face_id}.jpg")
        if os.path.exists(image_path):
            import shutil
            shutil.copy2(image_path, target_path)
        face_data = {
            "id": face_id,
            "name": name,
            "image_path": target_path,
            "position": position,
            "notes": notes,
            "created": datetime.now().isoformat(),
            "last_seen": None
        }
        self.faces[face_id] = face_data
        self.save_database()
        return face_id

    def add_plate(self, plate_number, image_path, owner="", vehicle_type="", notes=""):
        plate_id = hashlib.md5(f"{plate_number}_{datetime.now().timestamp()}".encode()).hexdigest()[:12]
        target_path = os.path.join(self.config['database']['plates_dir'], f"{plate_id}.jpg")
        if os.path.exists(image_path):
            import shutil
            shutil.copy2(image_path, target_path)
        plate_data = {
            "id": plate_id,
            "plate_number": plate_number,
            "image_path": target_path,
            "owner": owner,
            "vehicle_type": vehicle_type,
            "notes": notes,
            "created": datetime.now().isoformat(),
            "last_seen": None
        }
        self.plates[plate_id] = plate_data
        self.save_database()
        return plate_id

    def add_material(self, material_name, image_path, description=""):
        material_id = hashlib.md5(f"{material_name}_{datetime.now().timestamp()}".encode()).hexdigest()[:12]
        target_path = os.path.join(self.config['database'].get('materials_dir', 'database/materials'), f"{material_id}.jpg")
        if os.path.exists(image_path):
            import shutil
            shutil.copy2(image_path, target_path)
        material_data = {
            "id": material_id,
            "name": material_name,
            "image_path": target_path,
            "description": description,
            "created": datetime.now().isoformat(),
            "last_seen": None
        }
        self.materials[material_id] = material_data
        self.save_database()
        return material_id

    def recognize_face(self, face_image):
        return None, 0.0

class ModernObjectDetector:
    def __init__(self, config, database, model_manager, detection_type='yolo', detection_level='medium', triggers=None):
        self.config = config
        self.database = database
        self.model_manager = model_manager
        self.detection_type = detection_type
        self.detection_level = detection_level
        self.triggers = triggers or {
            'person': True,
            'vehicle': True,
            'motion': True
        }
        self.device_info = model_manager.get_device_info() if model_manager else {}
        self.yolo_version = model_manager.get_yolo_version() if model_manager else None
        self.yolo_available = model_manager.is_yolo_available() if model_manager else False
        if detection_type == 'yolo' and not self.yolo_available:
            print("ВНИМАНИЕ: YOLO не доступен, переключение на детектор движения")
            self.detection_type = 'motion_only'
        self.init_detector()
        self.next_object_id = 1
        self.colors = {
            'person': (0, 255, 0),
            'face': (0, 200, 0),
            'car': (255, 0, 0),
            'vehicle': (255, 100, 0),
            'bus': (255, 255, 0),
            'truck': (0, 255, 255),
            'motorcycle': (255, 0, 255),
            'plate': (0, 165, 255),
            'animal': (255, 165, 0),
            'bicycle': (128, 0, 128),
            'license_plate': (0, 255, 255),
            'material': (255, 192, 203),  # Розовый цвет для материалов
            'песок': (255, 182, 193),  # Светло-розовый для песка
            'щебень': (255, 105, 180),  # Ярко-розовый для щебня
            'гравий': (219, 112, 147),  # Бледно-фиолетовый для гравия
            'глина': (199, 21, 133),    # Средне-фиолетовый для глины
            'грунт': (218, 112, 214),   # Орхидея для грунта
            'асфальт': (186, 85, 211),  # Средне-фиолетовый для асфальта
            'бетон': (148, 0, 211)      # Темно-фиолетовый для бетона
        }
        self.level_settings = {
            'low': {
                'confidence_threshold': 0.7,
                'min_area': 1000,
                'face_recognition': False,
                'tracking': False
            },
            'medium': {
                'confidence_threshold': 0.5,
                'min_area': 500,
                'face_recognition': False,
                'tracking': True
            },
            'high': {
                'confidence_threshold': 0.3,
                'min_area': 200,
                'face_recognition': False,
                'tracking': True
            },
            'maximum': {
                'confidence_threshold': 0.2,
                'min_area': 100,
                'face_recognition': False,
                'tracking': True
            }
        }
        self.settings = self.level_settings.get(detection_level, self.level_settings['medium'])
        self.anpr_enabled = config.get('anpr', {}).get('enabled', False)
        self.use_separate_yolo = config.get('anpr', {}).get('use_separate_yolo', True)
        self.anpr_pipeline = None
        if self.anpr_enabled and model_manager:
            self.anpr_pipeline = model_manager.get_anpr_pipeline()
            if self.anpr_pipeline and self.anpr_pipeline.enabled:
                detector_loaded = self.anpr_pipeline.detector is not None and self.anpr_pipeline.detector.model is not None
                recognizer_loaded = self.anpr_pipeline.recognizer is not None and self.anpr_pipeline.recognizer.model is not None
                print(f"✅ ANPR система активирована: детектор={detector_loaded}, распознаватель={recognizer_loaded}")
            else:
                print("❌ ANPR система не доступна")
                self.anpr_enabled = False
        
        # Настройки для детекции материалов
        self.materials_enabled = config.get('materials', {}).get('enabled', True)
        self.use_materials_yolo = config.get('materials', {}).get('use_yolo', True)
        self.materials_list = config.get('materials', {}).get('materials_list', [])
        self.material_detection_threshold = config.get('materials', {}).get('detection_threshold', 0.5)
        self.materials_yolo_model = None
        
        # Инициализация модели YOLO для материалов
        if self.materials_enabled and self.use_materials_yolo and model_manager:
            self.materials_yolo_model = model_manager.get_materials_yolo_model()
            if self.materials_yolo_model:
                print(f"✅ YOLO модель для материалов загружена")
            else:
                print(f"❌ YOLO модель для материалов не доступна")
                self.use_materials_yolo = False
        
        self.stats = {
            'frames_processed': 0,
            'persons_detected': 0,
            'faces_detected': 0,
            'vehicles_detected': 0,
            'plates_detected': 0,
            'materials_detected': 0,
            'recognized_faces': 0,
            'recognized_plates': 0,
            'recognized_materials': 0,
            'device_info': self.device_info.get('device_key', 'cpu'),
            'yolo_version': self.yolo_version,
            'yolo_available': self.yolo_available,
            'anpr_enabled': self.anpr_enabled,
            'materials_enabled': self.materials_enabled,
            'materials_yolo_available': self.materials_yolo_model is not None
        }
        
        self.motion_buffer = deque(maxlen=5)
        self.trigger_stats = {'person': 0, 'vehicle': 0, 'motion': 0, 'ignored': 0}
        self.active_event = None
        self.event_frames = []
        self.event_start_time = None
        self.event_end_time = None
        self.save_path = None
        self.frame_counter = 0
        self.save_every_n_frame = 5
        self.max_frames_per_event = 50
        self.yolo_model = None
        self.last_processed_time = 0
        self.frame_skip_counter = 0
        self.frame_skip_factor = 1
        self.ui_update_interval = 33
        self._stats_cache = None
        self._stats_cache_time = 0
        self._frame_dimensions = None
        self._area_threshold = self.settings['min_area']
        self.init_models()

    def init_models(self):
        if HAS_YOLO and self.detection_type in ['yolo', 'auto']:
            self.yolo_model = self.model_manager.get_yolo_model() if self.model_manager else None
            if self.yolo_model:
                self.yolo_available = True

    def init_detector(self):
        with SuppressOutput():
            self.bg_subtractor = cv2.createBackgroundSubtractorMOG2(
                history=500, varThreshold=16, detectShadows=True)

    def set_save_path(self, save_path):
        self.save_path = save_path
        if save_path:
            os.makedirs(save_path, exist_ok=True)

    def set_save_settings(self, save_every_n_frame=None, max_frames_per_event=None):
        if save_every_n_frame is not None:
            self.save_every_n_frame = save_every_n_frame
        if max_frames_per_event is not None:
            self.max_frames_per_event = max_frames_per_event

    def set_frame_skip_factor(self, factor):
        self.frame_skip_factor = max(1, factor)

    def detect_yolo(self, frame):
        if not self.yolo_model or not self.yolo_available:
            print("YOLO модель не доступна, используем детектор движения")
            return self.detect_motion_only(frame)
        try:
            device = 'cpu'
            if HAS_TORCH and torch.cuda.is_available():
                device = '0' if self.device_info.get('device_key', 'cpu') in ['nvidia_gpu', 'amd_gpu'] else 'cpu'
            with SuppressOutput():
                results = self.yolo_model(frame, verbose=False, device=device)
            if results and len(results) > 0:
                return self._process_yolo_result(results[0], frame)
        except Exception as e:
            print(f"Ошибка YOLO детекции: {e}")
        return []

    def _process_yolo_result(self, yolo_result, frame):
        results = []
        try:
            if hasattr(yolo_result, 'boxes') and yolo_result.boxes is not None:
                boxes = yolo_result.boxes
                cls_ids = boxes.cls.cpu().numpy() if boxes.cls is not None else []
                confs = boxes.conf.cpu().numpy() if boxes.conf is not None else []
                xyxy = boxes.xyxy.cpu().numpy() if boxes.xyxy is not None else []
                class_names = []
                if hasattr(yolo_result, 'names'):
                    class_names = yolo_result.names
                else:
                    class_names = [
                        'person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train', 'truck', 'boat',
                        'traffic light', 'fire hydrant', 'stop sign', 'parking meter', 'bench', 'bird', 'cat',
                        'dog', 'horse', 'sheep', 'cow', 'elephant', 'bear', 'zebra', 'giraffe', 'backpack',
                        'umbrella', 'handbag', 'tie', 'suitcase', 'frisbee', 'skis', 'snowboard', 'sports ball',
                        'kite', 'baseball bat', 'baseball glove', 'skateboard', 'surfboard', 'tennis racket',
                        'bottle', 'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple',
                        'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza', 'donut', 'cake', 'chair',
                        'couch', 'potted plant', 'bed', 'dining table', 'toilet', 'tv', 'laptop', 'mouse', 'remote',
                        'keyboard', 'cell phone', 'microwave', 'oven', 'toaster', 'sink', 'refrigerator', 'book',
                        'clock', 'vase', 'scissors', 'teddy bear', 'hair drier', 'toothbrush'
                    ]
                for i in range(len(xyxy)):
                    if i < len(cls_ids) and i < len(confs):
                        x1, y1, x2, y2 = xyxy[i]
                        confidence = float(confs[i])
                        if confidence < self.settings['confidence_threshold']:
                            continue
                        class_id = int(cls_ids[i])
                        if class_id < len(class_names):
                            class_name = class_names[class_id]
                        else:
                            class_name = f"class_{class_id}"
                        x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
                        w = x2 - x1
                        h = y2 - y1
                        area = w * h
                        if area < self._area_threshold:
                            continue
                        object_type = 'other'
                        if class_name == 'person':
                            object_type = 'person'
                            self.stats['persons_detected'] += 1
                        elif class_name in ['car', 'bus', 'truck', 'motorcycle', 'bicycle']:
                            object_type = 'vehicle'
                            self.stats['vehicles_detected'] += 1
                        elif class_name in ['cat', 'dog', 'horse', 'sheep', 'cow', 'bird', 'elephant', 'bear', 'zebra', 'giraffe']:
                            object_type = 'animal'
                        elif class_name == 'face':
                            object_type = 'face'
                            self.stats['faces_detected'] += 1
                        elif 'license' in class_name.lower() or 'plate' in class_name.lower():
                            object_type = 'license_plate'
                            self.stats['plates_detected'] += 1
                        if (object_type == 'person' and self.triggers.get('person', True)) or \
                           (object_type == 'vehicle' and self.triggers.get('vehicle', True)) or \
                           (object_type == 'animal' and self.triggers.get('motion', True)) or \
                           (object_type == 'license_plate' and self.triggers.get('vehicle', True)) or \
                           object_type in ['face', 'other']:
                            detection = {
                                'class': object_type,
                                'class_name': class_name,
                                'confidence': confidence,
                                'bbox': [x1, y1, w, h],
                                'bbox_xyxy': [x1, y1, x2, y2],
                                'center': (x1 + w//2, y1 + h//2),
                                'area': area,
                                'is_yolo': True
                            }
                            results.append(detection)
                        if object_type == 'person':
                            self.trigger_stats['person'] += 1
                        elif object_type == 'vehicle' or object_type == 'license_plate':
                            self.trigger_stats['vehicle'] += 1
                        elif object_type in ['animal', 'other'] and self.triggers.get('motion', True):
                            self.trigger_stats['motion'] += 1
                        else:
                            self.trigger_stats['ignored'] += 1
        except Exception as e:
            print(f"Ошибка обработки YOLО результата: {str(e)}")
        return results

    def detect_motion_only(self, frame):
        if not self.triggers.get('motion', True):
            return []
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        blurred = cv2.GaussianBlur(gray, (21, 21), 0)
        fg_mask = self.bg_subtractor.apply(blurred)
        thresh = cv2.threshold(fg_mask, 25, 255, cv2.THRESH_BINARY)[1]
        thresh = cv2.dilate(thresh, None, iterations=2)
        contours, _ = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL,
                                      cv2.CHAIN_APPROX_SIMPLE)
        motion_objects = []
        for contour in contours:
            area = cv2.contourArea(contour)
            if area < self._area_threshold:
                continue
            x, y, w, h = cv2.boundingRect(contour)
            aspect_ratio = w / h if h > 0 else 0
            if 0.5 < aspect_ratio < 2.0 and area > 2000:
                object_type = 'person'
                self.stats['persons_detected'] += 1
                self.trigger_stats['person'] += 1
            elif 1.5 < aspect_ratio < 4.0 and area > 3000:
                object_type = 'vehicle'
                self.stats['vehicles_detected'] += 1
                self.trigger_stats['vehicle'] += 1
            else:
                object_type = 'other'
                if self.triggers.get('motion', True):
                    self.trigger_stats['motion'] += 1
                else:
                    self.trigger_stats['ignored'] += 1
                continue
            if (object_type == 'person' and not self.triggers.get('person', True)) or \
               (object_type == 'vehicle' and not self.triggers.get('vehicle', True)):
                continue
            motion_objects.append({
                'class': object_type,
                'class_name': object_type,
                'confidence': 0.6,
                'bbox': [x, y, w, h],
                'center': (x + w//2, y + h//2),
                'area': area,
                'from_motion': True
            })
        has_motion = len(motion_objects) > 0
        self.motion_buffer.append(has_motion)
        motion_stable = sum(self.motion_buffer) / len(self.motion_buffer) > 0.6
        if motion_stable:
            return motion_objects
        else:
            return []

    def detect_material_with_confidence(self, frame, vehicle_bbox):
        """Определяет материал в области транспортного средства с использованием YOLO v26"""
        if not self.materials_enabled:
            return None, 0.0
        
        x, y, w, h = vehicle_bbox
        height, width = frame.shape[:2]
        
        # Проверка границ
        x = max(0, min(x, width - 1))
        y = max(0, min(y, height - 1))
        w = max(1, min(w, width - x))
        h = max(1, min(h, height - y))
        
        roi = frame[y:y+h, x:x+w]
        if roi.size == 0:
            return None, 0.0
        
        try:
            # Используем YOLO модель для детекции материалов если доступна
            if self.use_materials_yolo and self.materials_yolo_model:
                return self._detect_material_with_yolo(roi)
            else:
                # Используем старую логику анализа текстуры и цвета
                return self._detect_material_with_texture(roi)
                
        except Exception as e:
            print(f"Ошибка определения материала: {e}")
            traceback.print_exc()
            return None, 0.0

    def _detect_material_with_yolo(self, roi):
        """Детекция материалов с использованием YOLO v26"""
        try:
            device = 'cpu'
            if HAS_TORCH and torch.cuda.is_available():
                device = '0' if self.device_info.get('device_key', 'cpu') in ['nvidia_gpu', 'amd_gpu'] else 'cpu'
            
            with SuppressOutput():
                results = self.materials_yolo_model(roi, verbose=False, device=device)
            
            if results and len(results) > 0 and hasattr(results[0], 'boxes'):
                boxes = results[0].boxes
                if boxes is not None and len(boxes) > 0:
                    cls_ids = boxes.cls.cpu().numpy() if boxes.cls is not None else []
                    confs = boxes.conf.cpu().numpy() if boxes.conf is not None else []
                    
                    # Ищем детекцию с максимальной уверенностью
                    best_idx = -1
                    best_confidence = 0.0
                    
                    for i in range(len(confs)):
                        if confs[i] > best_confidence and confs[i] >= self.material_detection_threshold:
                            best_confidence = confs[i]
                            best_idx = i
                    
                    if best_idx != -1:
                        class_id = int(cls_ids[best_idx])
                        # Получаем имя класса из модели
                        if hasattr(results[0], 'names'):
                            class_names = results[0].names
                            if class_id < len(class_names):
                                material_name = class_names[class_id]
                            else:
                                material_name = f"class_{class_id}"
                        else:
                            # Если нет имен классов, используем индекс
                            material_name = str(class_id)
                        
                        # Проверяем, есть ли это материал в нашем списке
                        if material_name in self.materials_list or any(mat in material_name.lower() for mat in self.materials_list):
                            return material_name, float(best_confidence)
                        else:
                            # Пытаемся сопоставить с нашим списком материалов
                            for material in self.materials_list:
                                if material.lower() in material_name.lower():
                                    return material, float(best_confidence)
            
            return None, 0.0
            
        except Exception as e:
            print(f"Ошибка YOLO детекции материалов: {e}")
            return None, 0.0

    def _detect_material_with_texture(self, roi):
        """Старая логика определения материалов по текстуре и цвету"""
        # Анализ текстуры и цвета
        gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        edges = cv2.Canny(blurred, 50, 150)
        
        # Анализ контуров
        contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            return None, 0.0
        
        # Расчет заполнения контурами
        h, w = roi.shape[:2]
        total_area = w * h
        contour_area = 0
        for contour in contours:
            contour_area += cv2.contourArea(contour)
        fill_ratio = contour_area / total_area if total_area > 0 else 0
        
        # Анализ цвета
        hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
        avg_hue = np.mean(hsv[:, :, 0])
        avg_saturation = np.mean(hsv[:, :, 1])
        avg_value = np.mean(hsv[:, :, 2])
        
        # Определение материала на основе характеристик
        material_confidence = 0.0
        material_name = None
        
        if fill_ratio > 0.3:
            if avg_saturation < 50:  # Ненасыщенные цвета
                if avg_value > 150:  # Светлые
                    material_name = "песок"
                    material_confidence = min(0.9, fill_ratio * 1.5)
                else:  # Темные
                    material_name = "гравий"
                    material_confidence = min(0.8, fill_ratio * 1.3)
            elif 20 < avg_hue < 40:  # Желто-коричневые оттенки
                material_name = "щебень"
                material_confidence = min(0.85, fill_ratio * 1.4)
            elif avg_hue > 100:  # Зеленоватые оттенки
                material_name = "глина"
                material_confidence = min(0.75, fill_ratio * 1.2)
            elif avg_saturation > 100:  # Высокая насыщенность
                material_name = "грунт"
                material_confidence = min(0.7, fill_ratio * 1.1)
        
        # Если материал не определен, но есть заполнение
        if not material_name and fill_ratio > 0.2:
            # Случайный выбор из списка материалов с низкой уверенностью
            material_name = random.choice(self.materials_list) if self.materials_list else None
            material_confidence = 0.3 + random.uniform(0.0, 0.2)
        
        # Проверка порога уверенности
        if material_name and material_confidence >= self.material_detection_threshold:
            return material_name, material_confidence
        else:
            return None, 0.0

    def filter_by_triggers(self, detections):
        if self.triggers.get('person', True) and self.triggers.get('vehicle', True) and self.triggers.get('motion', True):
            return detections
        filtered = []
        for det in detections:
            class_type = det['class']
            from_motion = det.get('from_motion', False)
            if class_type in ['face', 'person'] and self.triggers.get('person', True):
                filtered.append(det)
            elif class_type == 'vehicle' and self.triggers.get('vehicle', True):
                filtered.append(det)
            elif (class_type in ['animal', 'other'] or from_motion) and self.triggers.get('motion', True):
                filtered.append(det)
        return filtered

    def track_objects(self, detections):
        if not self.settings.get('tracking', True) or not detections:
            for det in detections:
                det['track_id'] = None
            return detections
        return detections

    def save_event_frames(self, event_type, frame, stream_name, results):
        if not self.save_path:
            return
        try:
            self.frame_counter += 1
            if self.frame_counter % self.save_every_n_frame != 0:
                return
            saved_frames_count = self.frame_counter // self.save_every_n_frame
            if saved_frames_count > self.max_frames_per_event:
                return
            current_time = datetime.now()
            hour_folder = current_time.strftime("%Y-%m-d_%H")
            camera_folder_name = stream_name.replace(" ", "_").replace("/", "_").replace("\\", "_")
            event_folder = os.path.join(self.save_path, camera_folder_name, hour_folder)
            os.makedirs(event_folder, exist_ok=True)
            frame_time = current_time.strftime("%Y-%m-d_%H-%M-%S-%f")[:-3]
            filename = os.path.join(event_folder, f"{frame_time}.jpg")
            cv2.imwrite(filename, frame)
            log_file = os.path.join(event_folder, "detection_log.txt")
            persons_count = len(results.get('persons', []))
            faces_count = len(results.get('faces', []))
            vehicles_count = len(results.get('vehicles', []))
            plates_count = len(results.get('plates', []))
            materials_count = len(results.get('materials', []))
            recognized_faces = results.get('recognized_faces', [])
            recognized_plates = results.get('recognized_plates', [])
            recognized_materials = results.get('recognized_materials', [])
            log_entry = f"Время: {current_time.strftime('%Y-%m-d %H:%M:%S.%f')[:-3]}\n"
            log_entry += f"Тип события: {event_type}\n"
            log_entry += f"Определено людей: {persons_count}\n"
            log_entry += f"Определено машин: {vehicles_count}\n"
            log_entry += f"Лиц определено: {faces_count}\n"
            if recognized_faces:
                log_entry += "Распознанные лица:\n"
                for face in recognized_faces:
                    if 'recognized' in face:
                        log_entry += f"  - {face['recognized']['name']} (уверенность: {face.get('recognition_confidence', 0):.2f})\n"
            log_entry += f"Номеров определено: {plates_count}\n"
            if recognized_plates:
                log_entry += "Распознанные номера (ANPR):\n"
                for plate in recognized_plates:
                    if 'plate_text' in plate:
                        plate_text = plate['plate_text']
                        confidence = plate.get('anpr_confidence', plate.get('confidence', 0))
                        bbox = plate.get('bbox', [0, 0, 0, 0])
                        w, h = bbox[2], bbox[3] if len(bbox) >= 4 else (0, 0)
                        log_entry += f"  - {plate_text} (размер: {w}x{h}, уверенность: {confidence:.2f})\n"
            log_entry += f"Материалов определено: {materials_count}\n"
            if recognized_materials:
                log_entry += "Распознанные материалы:\n"
                for material in recognized_materials:
                    if 'material_name' in material:
                        material_name = material['material_name']
                        confidence = material.get('confidence', 0)
                        vehicle_type = material.get('vehicle_type', 'грузовик')
                        bbox = material.get('bbox', [0, 0, 0, 0])
                        w, h = bbox[2], bbox[3] if len(bbox) >= 4 else (0, 0)
                        log_entry += f"  - {material_name} (уверенность: {confidence:.2f}) в {vehicle_type} (размер: {w}x{h})\n"
            log_entry += "-" * 50 + "\n"
            with open(log_file, 'a', encoding='utf-8') as f:
                f.write(log_entry)
            print(f"✅ Сохранен кадр с распознанным номером: {filename}")
        except Exception as e:
            print(f"Ошибка сохранения кадра: {e}")

    def process_frame(self, frame, stream_id=None, stream_name="Unknown"):
        current_time = time.time()
        if self._frame_dimensions is None:
            self._frame_dimensions = frame.shape[:2]
        self.stats['frames_processed'] += 1
        self.last_processed_time = current_time
        
        detections = []
        if self.detection_type == 'yolo':
            detections = self.detect_yolo(frame)
        elif self.detection_type == 'motion_only':
            detections = self.detect_motion_only(frame)
        else:
            detections = self.detect_motion_only(frame)
        
        # Детекция номеров ANPR
        anpr_detections = []
        if self.anpr_enabled and self.anpr_pipeline and self.anpr_pipeline.detector and self.anpr_pipeline.detector.model is not None and self.use_separate_yolo:
            anpr_detections = self.anpr_pipeline.detect_plates(frame)
            if anpr_detections:
                for det in anpr_detections:
                    det['from_anpr'] = True
                detections.extend(anpr_detections)
        
        # Фильтрация по триггерам
        filtered_detections = []
        for det in detections:
            class_type = det['class']
            if class_type == 'person' and self.triggers.get('person', True):
                filtered_detections.append(det)
            elif class_type == 'vehicle' and self.triggers.get('vehicle', True):
                filtered_detections.append(det)
            elif (class_type in ['animal', 'other'] or det.get('from_motion', False)) and self.triggers.get('motion', True):
                filtered_detections.append(det)
            elif class_type == 'face' and self.triggers.get('person', True):
                filtered_detections.append(det)
            elif class_type == 'license_plate' and self.triggers.get('vehicle', True):
                filtered_detections.append(det)
        detections = filtered_detections
        
        # Отслеживание объектов
        detections = self.track_objects(detections)
        
        # Распознавание номеров ANPR
        if self.anpr_enabled and self.anpr_pipeline and self.anpr_pipeline.recognizer and self.anpr_pipeline.recognizer.model is not None:
            plate_detections = [d for d in detections if d.get('class') == 'license_plate']
            if plate_detections:
                processed_plates = self.anpr_pipeline.process_detections(frame, plate_detections)
                plate_mapping = {}
                for i, det in enumerate(detections):
                    if det.get('class') == 'license_plate':
                        if 'bbox' in det and len(det['bbox']) == 4:
                            x, y, w, h = det['bbox']
                            center_key = f"{x + w//2},{y + h//2}"
                            plate_mapping[center_key] = i
                for processed in processed_plates:
                    if 'bbox' in processed and len(processed['bbox']) == 4:
                        x, y, w, h = processed['bbox']
                        center_key = f"{x + w//2},{y + h//2}"
                        if center_key in plate_mapping:
                            idx = plate_mapping[center_key]
                            detections[idx] = processed
                            if processed.get('plate_text'):
                                self.stats['recognized_plates'] += 1
                                print(f"✅ Распознан номер в процессе обработки: {processed['plate_text']}")
        
        # Инициализация результатов
        results = {
            'persons': [],
            'faces': [],
            'vehicles': [],
            'plates': [],
            'materials': [],
            'recognized_faces': [],
            'recognized_plates': [],
            'recognized_materials': [],
            'has_detections': False,
            'detection_type': None,
            'device_info': self.device_info,
            'yolo_version': self.yolo_version,
            'yolo_available': self.yolo_available,
            'anpr_enabled': self.anpr_enabled,
            'materials_enabled': self.materials_enabled,
            'materials_yolo_available': self.materials_yolo_model is not None,
            'trigger_stats': self.trigger_stats.copy()
        }
        
        has_detections = False
        materials_detections = []  # Отдельный список для детекций материалов
        
        # Обработка детекций
        for det in detections:
            class_type = det['class']
            
            # Обработка лиц и людей
            if class_type in ['face', 'person']:
                if class_type == 'face' and self.settings.get('face_recognition', False):
                    if 'bbox' in det and len(det['bbox']) >= 4:
                        x, y, w, h = det['bbox']
                        x, y, w, h = int(x), int(y), int(w), int(h)
                        height, width = frame.shape[:2]
                        x = max(0, min(x, width - 1))
                        y = max(0, min(y, height - 1))
                        w = max(1, min(w, width - x))
                        h = max(1, min(h, height - y))
                        face_roi = frame[y:y+h, x:x+w]
                        if face_roi.size > 0:
                            recognized_face, confidence = self.database.recognize_face(face_roi)
                            if recognized_face and confidence > 0.6:
                                det['recognized'] = recognized_face
                                det['recognition_confidence'] = confidence
                                results['recognized_faces'].append(det)
                                self.stats['recognized_faces'] += 1
                                results['detection_type'] = 'face_recognized'
                if class_type == 'face':
                    results['faces'].append(det)
                else:
                    results['persons'].append(det)
                has_detections = True
                
            # Обработка транспортных средств и номеров
            elif class_type == 'vehicle' or class_type == 'license_plate':
                if class_type == 'license_plate':
                    results['plates'].append(det)
                    if 'plate_text' in det and det['plate_text']:
                        results['recognized_plates'].append(det)
                        self._create_plate_event(det, stream_id, stream_name)
                else:
                    results['vehicles'].append(det)
                    
                    # Детекция материалов в транспортных средствах
                    if self.materials_enabled and 'bbox' in det and len(det['bbox']) >= 4:
                        material_name, material_confidence = self.detect_material_with_confidence(frame, det['bbox'])
                        if material_name and material_confidence >= self.material_detection_threshold:
                            # Создание отдельной детекции для материала
                            material_detection = {
                                'class': 'material',
                                'class_name': 'material',
                                'material_name': material_name,
                                'confidence': material_confidence,
                                'bbox': det['bbox'].copy(),  # Используем ту же область
                                'vehicle_type': det.get('class_name', 'транспорт'),
                                'vehicle_bbox': det['bbox'],
                                'associated_vehicle': det
                            }
                            materials_detections.append(material_detection)
                            
                            # Добавляем информацию о материале в детекцию транспортного средства
                            det['material'] = material_name
                            det['material_confidence'] = material_confidence
                            
                            # Добавляем в результаты
                            results['materials'].append(material_detection)
                            results['recognized_materials'].append(material_detection)
                            self.stats['materials_detected'] += 1
                            self.stats['recognized_materials'] += 1
                            print(f"✅ Обнаружен материал: {material_name} (уверенность: {material_confidence:.2f}) в {det.get('class_name', 'транспорте')}")
                
                has_detections = True
                
            # Обработка движения и других объектов
            elif det.get('from_motion', False) or class_type in ['animal', 'other']:
                has_detections = True
        
        # Отрисовка всех детекций
        self._draw_detections(frame, detections, materials_detections, results)
        
        results['has_detections'] = has_detections
        
        # Обработка событий
        if has_detections and results.get('recognized_plates'):
            if not self.active_event:
                self.active_event = 'plate_recognized'
                self.event_start_time = datetime.now()
                self.save_event_frames(self.active_event, frame.copy(), stream_name, results)
            else:
                self.save_event_frames(self.active_event, frame.copy(), stream_name, results)
                self.event_end_time = datetime.now()
        elif has_detections and results.get('recognized_materials'):
            if not self.active_event:
                self.active_event = 'material_recognized'
                self.event_start_time = datetime.now()
                self.save_event_frames(self.active_event, frame.copy(), stream_name, results)
            else:
                self.save_event_frames(self.active_event, frame.copy(), stream_name, results)
                self.event_end_time = datetime.now()
        elif has_detections:
            if not self.active_event:
                self.active_event = results.get('detection_type', 'object_detected')
                self.event_start_time = datetime.now()
                self.save_event_frames(self.active_event, frame.copy(), stream_name, results)
            else:
                self.save_event_frames(self.active_event, frame.copy(), stream_name, results)
                self.event_end_time = datetime.now()
        else:
            if self.active_event:
                self.active_event = None
                self.event_start_time = None
                self.event_end_time = None
        
        results['triggers'] = self.triggers
        self.draw_stats(frame)
        
        return frame, results

    def _draw_detections(self, frame, detections, materials_detections, results):
        """Отрисовка всех типов детекций на кадре"""
        # Отрисовка стандартных детекций (лица, люди, транспорт)
        for det in detections:
            class_type = det['class']
            
            if class_type in ['face', 'person']:
                self._draw_person_detection(frame, det)
            elif class_type == 'vehicle' or class_type == 'license_plate':
                self._draw_vehicle_detection(frame, det)
            elif det.get('from_motion', False) or class_type in ['animal', 'other']:
                self._draw_motion_detection(frame, det)
        
        # Отрисовка детекций материалов отдельно
        for material_det in materials_detections:
            self._draw_material_detection(frame, material_det)

    def _draw_person_detection(self, frame, det):
        """Отрисовка детекции лица или человека"""
        if 'bbox' in det and len(det['bbox']) >= 4:
            x, y, w, h = det['bbox']
            x, y, w, h = int(x), int(y), int(w), int(h)
            height, width = frame.shape[:2]
            x = max(0, min(x, width - 1))
            y = max(0, min(y, height - 1))
            w = max(1, min(w, width - x))
            h = max(1, min(h, height - y))
            
            color = self.colors.get(det['class'], (0, 255, 0))
            cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
            
            label = f"{det.get('class_name', det['class'])}: {det['confidence']:.2f}"
            if 'recognized' in det:
                label = f"{det['recognized']['name']} ({det['recognition_confidence']:.2f})"
            elif 'plate_text' in det:
                label = f"Номер: {det['plate_text']}"
            
            text_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)[0]
            text_x = max(0, x)
            text_y = max(0, y - text_size[1] - 5)
            text_w = text_size[0]
            text_h = text_size[1] + 5
            
            cv2.rectangle(frame, (text_x, text_y),
                         (text_x + text_w, text_y + text_h), color, -1)
            cv2.putText(frame, label, (text_x, text_y + text_h - 5),
                      cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)

    def _draw_vehicle_detection(self, frame, det):
        """Отрисовка детекции транспортного средства или номера"""
        if 'bbox' in det and len(det['bbox']) >= 4:
            x, y, w, h = det['bbox']
            x, y, w, h = int(x), int(y), int(w), int(h)
            height, width = frame.shape[:2]
            x = max(0, min(x, width - 1))
            y = max(0, min(y, height - 1))
            w = max(1, min(w, width - x))
            h = max(1, min(h, height - y))
            
            # Определение цвета
            if det['class'] == 'license_plate':
                color = self.colors.get('license_plate', (0, 255, 255))
                thickness = 1
            else:
                color = self.colors.get('vehicle', (255, 0, 0))
                thickness = 2
            
            cv2.rectangle(frame, (x, y), (x + w, y + h), color, thickness)
            
            # Формирование подписи
            label = f"{det.get('class_name', 'License Plate' if det['class'] == 'license_plate' else 'Vehicle')}: {det['confidence']:.2f}"
            if 'plate_text' in det:
                label = f"Номер: {det['plate_text']} ({w}x{h})"
            if 'material' in det:
                label = f"{label} | Мат: {det['material']} ({det.get('material_confidence', 0):.2f})"
            
            text_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)[0]
            text_x = max(0, x)
            text_y = max(0, y - text_size[1] - 5)
            text_w = text_size[0]
            text_h = text_size[1] + 5
            
            cv2.rectangle(frame, (text_x, text_y),
                         (text_x + text_w, text_y + text_h), color, -1)
            cv2.putText(frame, label, (text_x, text_y + text_h - 5),
                      cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)

    def _draw_material_detection(self, frame, det):
        """Отрисовка детекции материала отдельным боксом"""
        if 'bbox' in det and len(det['bbox']) >= 4:
            x, y, w, h = det['bbox']
            x, y, w, h = int(x), int(y), int(w), int(h)
            height, width = frame.shape[:2]
            x = max(0, min(x, width - 1))
            y = max(0, min(y, height - 1))
            w = max(1, min(w, width - x))
            h = max(1, min(h, height - y))
            
            # Получаем цвет для конкретного материала или используем цвет по умолчанию
            material_name = det.get('material_name', 'material')
            if material_name in self.colors:
                color = self.colors[material_name]
            else:
                color = self.colors.get('material', (255, 192, 203))
            
            thickness = 2
            
            # Создаем смещенный бокс для материала (немного выше транспортного средства)
            material_y = max(0, y - 25)
            material_h = 20  # Высота бокса материала
            
            # Отрисовка бокса материала
            cv2.rectangle(frame, (x, material_y), (x + w, material_y + material_h), color, thickness)
            
            # Формирование подписи с процентом уверенности
            confidence = det.get('confidence', 0)
            label = f"{material_name}: {confidence:.0%}"
            
            text_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)[0]
            text_x = max(0, x)
            text_y = max(0, material_y - text_size[1] - 5)
            text_w = text_size[0]
            text_h = text_size[1] + 5
            
            # Фон для текста
            cv2.rectangle(frame, (text_x, text_y),
                         (text_x + text_w, text_y + text_h), color, -1)
            
            # Текст с процентом уверенности
            cv2.putText(frame, label, (text_x, text_y + text_h - 5),
                      cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2)

    def _draw_motion_detection(self, frame, det):
        """Отрисовка детекции движения"""
        if 'bbox' in det and len(det['bbox']) >= 4:
            x, y, w, h = det['bbox']
            x, y, w, h = int(x), int(y), int(w), int(h)
            height, width = frame.shape[:2]
            x = max(0, min(x, width - 1))
            y = max(0, min(y, height - 1))
            w = max(1, min(w, width - x))
            h = max(1, min(h, height - y))
            
            color = (255, 255, 0) if det.get('from_motion', False) else self.colors.get(det['class'], (255, 165, 0))
            cv2.rectangle(frame, (x, y), (x + w, y + h), color, 1)
            
            motion_label = f"Motion: {det['class']}" if det.get('from_motion', False) else f"{det['class']}"
            text_size = cv2.getTextSize(motion_label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)[0]
            text_x = max(0, x)
            text_y = max(0, y - text_size[1] - 2)
            text_w = text_size[0]
            text_h = text_size[1] + 2
            
            cv2.rectangle(frame, (text_x, text_y),
                         (text_x + text_w, text_y + text_h), color, -1)
            cv2.putText(frame, motion_label, (text_x, text_y + text_h - 2),
                      cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)

    def _create_plate_event(self, plate_detection, stream_id, stream_name):
        if 'plate_text' in plate_detection and plate_detection['plate_text']:
            print(f"🔔 Создано событие для номера: {plate_detection['plate_text']} (поток: {stream_name})")

    def draw_stats(self, frame):
        current_time = time.time()
        if self._stats_cache is None or current_time - self._stats_cache_time > 1.0:
            yolo_ver = self.yolo_version or 'N/A'
            yolo_status = "✅" if self.yolo_available else "❌"
            save_status = "💾" if self.save_path else "🚫"
            anpr_status = "✅" if self.anpr_enabled else "🚫"
            materials_status = "✅" if self.materials_enabled else "🚫"
            materials_yolo_status = "✅" if self.materials_yolo_model is not None else "🚫"
            
            recognized_plates = self.stats.get('recognized_plates', 0)
            plates_detected = self.stats.get('plates_detected', 0)
            materials_detected = self.stats.get('materials_detected', 0)
            recognized_materials = self.stats.get('recognized_materials', 0)
            
            anpr_info = f"ANPR: {recognized_plates}/{plates_detected} распознано"
            materials_info = f"Мат: {recognized_materials}/{materials_detected} распознано"
            materials_yolo_info = f"YOLO мат: {materials_yolo_status}"
            
            self._stats_text = [
                f"YOLOv26: {yolo_ver} {yolo_status} | {anpr_info} | {materials_info} | {materials_yolo_info} | Save: {save_status}",
                f"Frames: {self.stats['frames_processed']}",
                f"Persons: {self.stats['persons_detected']} | Triggers: {self.trigger_stats.get('person', 0)}",
                f"Faces: {self.stats['faces_detected']}",
                f"Vehicles: {self.stats['vehicles_detected']} | Triggers: {self.trigger_stats.get('vehicle', 0)}",
                f"Motion: {self.stats.get('motion_detected', 0)} | Triggers: {self.trigger_stats.get('motion', 0)}",
                f"Plates: {plates_detected} | Recognized: {recognized_plates}",
                f"Materials: {materials_detected} | Recognized: {recognized_materials}",
                f"Save: {self.save_path or 'Not set'}"
            ]
            self._stats_cache_time = current_time
        
        y_offset = 20
        for text in self._stats_text:
            text_size = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)[0]
            cv2.rectangle(frame, (5, y_offset - text_size[1] - 2),
                         (5 + text_size[0], y_offset), (0, 0, 0), -1)
            cv2.putText(frame, text, (10, y_offset - 2),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)
            y_offset += 20

class VideoStreamThread(threading.Thread):
    def __init__(self, stream_config, event_callback, database, model_manager, save_path, save_every_n_frame=5, max_frames_per_event=50):
        super().__init__(daemon=True)
        self.stream_config = stream_config
        self.event_callback = event_callback
        self.database = database
        self.model_manager = model_manager
        self.save_path = save_path
        self.save_every_n_frame = save_every_n_frame
        self.max_frames_per_event = max_frames_per_event
        self.stream_id = stream_config['id']
        self.stream_name = stream_config['name']
        self.stream_url = stream_config['url']
        self.cell_index = stream_config.get('cell_index', 0)
        self.detection_type = stream_config.get('detection_type', 'yolo')
        self.detection_level = stream_config.get('detection_level', 'medium')
        self.triggers = stream_config.get('triggers', {
            'person': True,
            'vehicle': True,
            'motion': True
        })
        yolo_available = model_manager.is_yolo_available() if model_manager else False
        detector_config = model_manager.config if hasattr(model_manager, 'config') else {}
        self.detector = ModernObjectDetector(
            detector_config,
            database,
            model_manager,
            self.detection_type,
            self.detection_level,
            self.triggers
        )
        self.detector.set_save_path(self.save_path)
        self.detector.set_save_settings(self.save_every_n_frame, self.max_frames_per_event)
        self._running = False
        self._stop_event = threading.Event()
        self.cap = None
        self.frame_queue = queue.Queue(maxsize=2)
        self.processing_interval = 1.0 / stream_config.get('fps', 15)
        self.last_processed = 0
        self.frame_skip_factor = 1
        self.last_frame_time = 0
        self.frame_counter = 0
        device_info = model_manager.get_device_info() if model_manager else {}
        self.stats = {
            'status': 'initializing',
            'frames_processed': 0,
            'frames_skipped': 0,
            'last_event': None,
            'detection_type': self.detection_type,
            'detection_level': self.detection_level,
            'triggers': self.triggers,
            'device_info': device_info,
            'yolo_version': device_info.get('yolo_version', 'N/A'),
            'yolo_available': yolo_available,
            'anpr_enabled': detector_config.get('anpr', {}).get('enabled', False),
            'materials_enabled': detector_config.get('materials', {}).get('enabled', True),
            'materials_yolo_enabled': detector_config.get('materials', {}).get('use_yolo', True),
            'start_time': time.time()
        }
        self._lock = threading.RLock()
        self._initialized = threading.Event()

    def run(self):
        threading.current_thread().name = f"Stream-{self.stream_id}"
        self._running = True
        self._stop_event.clear()
        self.stats['status'] = 'connecting'
        try:
            if self.stream_url.startswith('onvif://') and HAS_ONVIF:
                try:
                    url_parts = self.stream_url[8:].split('@')
                    auth_part = url_parts[0]
                    host_part = url_parts[1] if len(url_parts) > 1 else ''
                    auth_parts = auth_part.split(':')
                    username = auth_parts[0] if auth_parts else 'admin'
                    password = auth_parts[1] if len(auth_parts) > 1 else '123456'
                    host_parts = host_part.split(':')
                    ip = host_parts[0] if host_parts else '192.168.1.100'
                    port = int(host_parts[1]) if len(host_parts) > 1 else 80
                    with SuppressOutput():
                        camera = ONVIFCamera(ip, port, username, password)
                        media_service = camera.create_media_service()
                        profiles = media_service.GetProfiles()
                        if profiles:
                            stream_uri = media_service.GetStreamUri({
                                'StreamSetup': {
                                    'Stream': 'RTP-Unicast',
                                    'Transport': {'Protocol': 'RTSP'}
                                },
                                'ProfileToken': profiles[0].token
                            })
                            rtsp_url = stream_uri.Uri
                            if username and password:
                                rtsp_url = rtsp_url.replace('rtsp://', f'rtsp://{username}:{password}@')
                            self.stream_url = rtsp_url
                except Exception as e:
                    self.stats['status'] = f'onvif_error: {str(e)[:50]}'
                    self._initialized.set()
                    return
            elif self.stream_url.startswith('onvif://'):
                self.stats['status'] = 'onvif_not_available'
                self._initialized.set()
                return
            self.cap = cv2.VideoCapture(self.stream_url)
            self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
            if not self.cap.isOpened():
                self.stats['status'] = 'connection_failed'
                self._initialized.set()
                return
            self.stats['status'] = 'running'
            self._initialized.set()
            target_fps = self.stream_config.get('fps', 15)
            frame_interval = 1.0 / target_fps
            frame_counter = 0
            while self._running and not self._stop_event.is_set():
                try:
                    frame_start = time.time()
                    ret, frame = self.cap.read()
                    if not ret:
                        self.handle_stream_loss()
                        time.sleep(0.5)
                        continue
                    frame_counter += 1
                    if frame_counter % self.frame_skip_factor != 0:
                        continue
                    self.process_frame(frame, time.time())
                    elapsed = time.time() - frame_start
                    sleep_time = max(0, frame_interval - elapsed)
                    if sleep_time > 0:
                        time.sleep(sleep_time)
                except Exception as e:
                    print(f"Ошибка в потоке {self.stream_name}: {e}")
                    time.sleep(0.5)
        except Exception as e:
            self.stats['status'] = f'error: {str(e)[:50]}'
        finally:
            self.cleanup()

    def set_frame_skip_factor(self, factor):
        self.frame_skip_factor = max(1, factor)
        if hasattr(self.detector, 'set_frame_skip_factor'):
            self.detector.set_frame_skip_factor(factor)

    def process_frame(self, frame, timestamp):
        try:
            processed_frame, results = self.detector.process_frame(frame, stream_id=self.stream_id, stream_name=self.stream_name)
            if not self.frame_queue.full():
                try:
                    self.frame_queue.put(processed_frame, timeout=0.01)
                except queue.Full:
                    pass
            if results.get('has_detections', False):
                self.create_modern_report(results)
                for face in results['recognized_faces']:
                    event = self.create_event('face_recognized', face)
                    self.event_callback(event)
                for plate in results.get('recognized_plates', []):
                    if 'plate_text' in plate:
                        event_data = {
                            'plate_text': plate['plate_text'],
                            'confidence': plate.get('anpr_confidence', plate.get('confidence', 0.5)),
                            'bbox': plate['bbox'],
                            'stream_info': {
                                'name': self.stream_name,
                                'id': self.stream_id
                            },
                            'timestamp': datetime.now().isoformat(),
                            'recognized_at': plate.get('recognized_at', datetime.now().isoformat())
                        }
                        event = self.create_event('plate_recognized', event_data)
                        self.event_callback(event)
                for material in results.get('recognized_materials', []):
                    if 'material_name' in material:
                        event_data = {
                            'material_name': material['material_name'],
                            'confidence': material.get('confidence', 0.5),
                            'vehicle_type': material.get('vehicle_type', 'грузовик'),
                            'bbox': material.get('bbox', []),
                            'stream_info': {
                                'name': self.stream_name,
                                'id': self.stream_id
                            },
                            'timestamp': datetime.now().isoformat()
                        }
                        event = self.create_event('material_recognized', event_data)
                        self.event_callback(event)
                if results['persons'] or results['faces']:
                    event_data = {
                        'persons': len(results['persons']),
                        'faces': len(results['faces']),
                        'vehicles': len(results['vehicles']),
                        'plates': len(results['plates']),
                        'materials': len(results.get('materials', [])),
                        'recognized_plates': len(results.get('recognized_plates', [])),
                        'recognized_materials': len(results.get('recognized_materials', [])),
                        'motion_detected': self.detector.stats.get('motion_detected', 0) > 0,
                        'triggers': self.triggers,
                        'trigger_stats': results.get('trigger_stats', {}),
                        'yolo_version': results.get('yolo_version', 'N/A'),
                        'yolo_available': results.get('yolo_available', False),
                        'anpr_enabled': results.get('anpr_enabled', False),
                        'materials_enabled': results.get('materials_enabled', False),
                        'materials_yolo_available': results.get('materials_yolo_available', False),
                        'detection_type': self.detection_type,
                        'detection_level': self.detection_level
                    }
                    event = self.create_event('person_detected', event_data)
                    self.event_callback(event)
                if results['vehicles'] or results['plates'] or results.get('materials', []):
                    event_data = {
                        'vehicles': len(results['vehicles']),
                        'plates': len(results['plates']),
                        'materials': len(results.get('materials', [])),
                        'recognized_plates': len(results.get('recognized_plates', [])),
                        'recognized_materials': len(results.get('recognized_materials', [])),
                        'persons': len(results['persons']),
                        'faces': len(results['faces']),
                        'motion_detected': self.detector.stats.get('motion_detected', 0) > 0,
                        'triggers': self.triggers,
                        'trigger_stats': results.get('trigger_stats', {}),
                        'yolo_version': results.get('yolo_version', 'N/A'),
                        'yolo_available': results.get('yolo_available', False),
                        'anpr_enabled': results.get('anpr_enabled', False),
                        'materials_enabled': results.get('materials_enabled', False),
                        'materials_yolo_available': results.get('materials_yolo_available', False),
                        'detection_type': self.detection_type,
                        'detection_level': self.detection_level
                    }
                    event = self.create_event('vehicle_detected', event_data)
                    self.event_callback(event)
                if results.get('trigger_stats', {}).get('motion', 0) > 0:
                    event_data = {
                        'motion_count': results.get('trigger_stats', {}).get('motion', 0),
                        'persons': len(results['persons']),
                        'vehicles': len(results['vehicles']),
                        'materials': len(results.get('materials', [])),
                        'triggers': self.triggers,
                        'trigger_stats': results.get('trigger_stats', {}),
                        'yolo_version': results.get('yolo_version', 'N/A'),
                        'yolo_available': results.get('yolo_available', False),
                        'anpr_enabled': results.get('anpr_enabled', False),
                        'materials_enabled': results.get('materials_enabled', False),
                        'materials_yolo_available': results.get('materials_yolo_available', False),
                        'detection_type': self.detection_type,
                        'detection_level': self.detection_level
                    }
                    event = self.create_event('motion_detected', event_data)
                    self.event_callback(event)
            with self._lock:
                self.stats['frames_processed'] += 1
                self.stats['last_event'] = datetime.now()
        except Exception as e:
            print(f"Ошибка обработки кадра: {e}")
            traceback.print_exc()

    def create_modern_report(self, results):
        current_time = datetime.now().strftime("%H:%M:%S")
        vehicles_count = len(results.get('vehicles', []))
        plates_count = len(results.get('plates', []))
        materials_count = len(results.get('materials', []))
        recognized_plates = results.get('recognized_plates', [])
        recognized_materials = results.get('recognized_materials', [])
        if vehicles_count > 0:
            report_lines = []
            report_lines.append(f"{current_time}, Обнаружен транспорт {vehicles_count} количество")
            if recognized_plates:
                for plate in recognized_plates:
                    if 'plate_text' in plate:
                        report_lines.append(f"{current_time}, у транспорта Обнаружен номер ({plate['plate_text']})")
            elif plates_count > 0:
                report_lines.append(f"{current_time}, у транспорта обнаружено {plates_count} номеров (не распознаны)")
            if recognized_materials:
                for material in recognized_materials:
                    if 'material_name' in material:
                        vehicle_type = material.get('vehicle_type', 'транспорте')
                        confidence = material.get('confidence', 0)
                        report_lines.append(f"{current_time}, в {vehicle_type} обнаружен материал ({material['material_name']}) уверенность {confidence:.0%}")
            elif materials_count > 0:
                report_lines.append(f"{current_time}, у транспорта обнаружено {materials_count} материалов")
            self.save_modern_report("\n".join(report_lines))
        persons_count = len(results.get('persons', []))
        faces_count = len(results.get('faces', []))
        if persons_count > 0:
            report_line = f"{current_time}, Обнаружен Человек {persons_count} количество"
            if faces_count > 0:
                report_line += f", Лиц обнаружено {faces_count}"
            self.save_modern_report(report_line)

    def save_modern_report(self, report_line):
        try:
            report_file = os.path.join(self.save_path, f"modern_report_{datetime.now().strftime('%Y-%m-d')}.txt")
            with open(report_file, 'a', encoding='utf-8') as f:
                f.write(report_line + "\n")
        except Exception as e:
            print(f"Ошибка сохранения отчета: {e}")

    def create_event(self, event_type, data):
        if isinstance(data, dict):
            data['triggers'] = self.triggers
        return {
            'stream_id': self.stream_id,
            'stream_name': self.stream_name,
            'event_type': event_type,
            'data': data,
            'timestamp': datetime.now(),
            'frame_time': datetime.now().strftime("%H:%M:%S.%f")[:-3],
            'detection_type': self.detection_type,
            'detection_level': self.detection_level,
            'triggers': self.triggers,
            'yolo_version': self.detector.yolo_version,
            'yolo_available': self.detector.yolo_available,
            'anpr_enabled': self.detector.anpr_enabled,
            'materials_enabled': self.detector.materials_enabled,
            'materials_yolo_available': self.detector.materials_yolo_model is not None,
            'cell_index': self.cell_index
        }

    def handle_stream_loss(self):
        if self._stop_event.is_set():
            return
        self.stats['status'] = 'reconnecting'
        time.sleep(2)
        if self._stop_event.is_set():
            return
        if self.cap:
            self.cap.release()
        try:
            self.cap = cv2.VideoCapture(self.stream_url)
            if self.cap.isOpened():
                self.stats['status'] = 'running'
            else:
                self.stats['status'] = 'connection_failed'
        except:
            self.stats['status'] = 'connection_failed'

    def get_frame(self):
        try:
            return self.frame_queue.get_nowait()
        except queue.Empty:
            return None

    def stop(self):
        if not self._running:
            return
        self._running = False
        self._stop_event.set()
        if not self._initialized.is_set():
            self._initialized.wait(timeout=2.0)
        if self.cap:
            self.cap.release()
            self.cap = None
        self.stats['status'] = 'stopped'
        while not self.frame_queue.empty():
            try:
                self.frame_queue.get_nowait()
            except queue.Empty:
                break

    def is_running(self):
        return self._running and not self._stop_event.is_set()

    def wait_for_initialization(self, timeout=5.0):
        return self._initialized.wait(timeout=timeout)

    def cleanup(self):
        self._running = False
        if self.cap:
            self.cap.release()
            self.cap = None
        self.stats['status'] = 'stopped'
        self._initialized.set()

class StreamSelectorWindow:
    def __init__(self, parent, streams, on_stream_selected):
        self.parent = parent
        self.streams = streams
        self.on_stream_selected = on_stream_selected
        self.result = None
        self.window = tk.Toplevel(parent)
        self.window.title("Управление потоками")
        self.window.geometry("800x500")
        self.window.resizable(True, True)
        self.bg_color = '#f0f2f5'
        self.panel_bg = '#ffffff'
        self.fg_color = '#1a1a1a'
        self.accent_color = '#3b82f6'
        self.success_color = '#10b981'
        self.window.configure(bg=self.bg_color)
        self.center_window()
        self.stream_data_dict = {}
        self.setup_ui()
        self.window.grab_set()

    def center_window(self):
        self.window.update_idletasks()
        width = self.window.winfo_width()
        height = self.window.winfo_height()
        x = (self.window.winfo_screenwidth() // 2) - (width // 2)
        y = (self.window.winfo_screenheight() // 2) - (height // 2)
        self.window.geometry(f'{width}x{height}+{x}+{y}')

    def setup_ui(self):
        top_frame = tk.Frame(self.window, bg=self.panel_bg, height=60)
        top_frame.pack(fill=tk.X, pady=(0, 10))
        top_frame.pack_propagate(False)
        title_label = tk.Label(top_frame, text="📹 Управление потоками",
                              font=('Segoe UI', 14, 'bold'),
                              bg=self.panel_bg, fg=self.fg_color)
        title_label.pack(side=tk.LEFT, padx=20, pady=10)
        button_frame = tk.Frame(top_frame, bg=self.panel_bg)
        button_frame.pack(side=tk.RIGHT, padx=20, pady=10)
        tk.Button(button_frame, text="➕ Добавить",
                 command=self.add_stream,
                 bg=self.accent_color, fg='white',
                 font=('Segoe UI', 9, 'bold'),
                 padx=8, pady=4).pack(side=tk.LEFT, padx=2)
        tk.Button(button_frame, text="✏️ Редактировать",
                 command=self.edit_stream,
                 bg=self.success_color, fg='white',
                 font=('Segoe UI', 9, 'bold'),
                 padx=8, pady=4).pack(side=tk.LEFT, padx=2)
        tk.Button(button_frame, text="🗑️ Удалить",
                 command=self.delete_stream,
                 bg='#ef4444', fg='white',
                 font=('Segoe UI', 9, 'bold'),
                 padx=8, pady=4).pack(side=tk.LEFT, padx=2)
        main_frame = tk.Frame(self.window, bg=self.bg_color)
        main_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        columns = ('ID', 'Название', 'IP адрес', 'Порт', 'Тип', 'Статус', 'Триггеры')
        self.tree = ttk.Treeview(main_frame, columns=columns, show='headings', height=15)
        for col in columns:
            self.tree.heading(col, text=col)
            self.tree.column(col, width=100)
        self.tree.column('ID', width=50)
        self.tree.column('Название', width=150)
        self.tree.column('IP адрес', width=120)
        self.tree.column('Порт', width=60)
        self.tree.column('Тип', width=80)
        self.tree.column('Статус', width=100)
        self.tree.column('Триггеры', width=120)
        scrollbar_y = ttk.Scrollbar(main_frame, orient=tk.VERTICAL, command=self.tree.yview)
        scrollbar_x = ttk.Scrollbar(main_frame, orient=tk.HORIZONTAL, command=self.tree.xview)
        self.tree.configure(yscrollcommand=scrollbar_y.set, xscrollcommand=scrollbar_x.set)
        self.tree.grid(row=0, column=0, sticky=tk.NSEW, padx=(0, 5), pady=(0, 5))
        scrollbar_y.grid(row=0, column=1, sticky=tk.NS, pady=(0, 5))
        scrollbar_x.grid(row=1, column=0, sticky=tk.EW, padx=(0, 5))
        main_frame.grid_rowconfigure(0, weight=1)
        main_frame.grid_columnconfigure(0, weight=1)
        self.tree.bind('<Double-Button-1>', self.on_double_click)
        self.load_streams()
        info_frame = tk.Frame(self.window, bg=self.panel_bg, height=80)
        info_frame.pack(fill=tk.X, pady=(10, 0))
        info_frame.pack_propagate(False)
        info_text = "Выберите поток для редактирования или дважды кликните для открытия в полноэкранном режиме"
        info_label = tk.Label(info_frame, text=info_text,
                             font=('Segoe UI', 9),
                             bg=self.panel_bg, fg=self.fg_color,
                             wraplength=700, justify=tk.LEFT)
        info_label.pack(padx=20, pady=20)
        bottom_frame = tk.Frame(self.window, bg=self.bg_color, height=50)
        bottom_frame.pack(fill=tk.X, side=tk.BOTTOM, pady=(10, 0))
        bottom_frame.pack_propagate(False)
        ok_button = tk.Button(bottom_frame, text="✅ ОК",
                            command=self.on_ok,
                            bg=self.success_color, fg='white',
                            font=('Segoe UI', 9, 'bold'),
                            padx=12, pady=6)
        ok_button.pack(side=tk.RIGHT, padx=(10, 20))
        save_button = tk.Button(bottom_frame, text="💾 Сохранить",
                              command=self.on_save,
                              bg=self.accent_color, fg='white',
                              font=('Segoe UI', 9, 'bold'),
                              padx=12, pady=6)
        save_button.pack(side=tk.RIGHT, padx=(0, 10))
        cancel_button = tk.Button(bottom_frame, text="❌ Отмена",
                                command=self.window.destroy,
                                bg='#ef4444', fg='white',
                                font=('Segoe UI', 9, 'bold'),
                                padx=12, pady=6)
        cancel_button.pack(side=tk.LEFT, padx=(20, 0))

    def load_streams(self):
        for item in self.tree.get_children():
            self.tree.delete(item)
        self.stream_data_dict.clear()
        for stream in self.streams:
            triggers = stream.get('triggers', {})
            trigger_text = []
            if triggers.get('person', False):
                trigger_text.append('👤')
            if triggers.get('vehicle', False):
                trigger_text.append('🚗')
            if triggers.get('motion', False):
                trigger_text.append('🎥')
            item_id = self.tree.insert('', 'end', values=(
                stream.get('id', ''),
                stream.get('name', ''),
                stream.get('ip', ''),
                stream.get('port', ''),
                stream.get('detection_type', 'yolo'),
                'Не активен',
                ' '.join(trigger_text) if trigger_text else 'Нет'
            ))
            self.stream_data_dict[item_id] = stream

    def get_selected_stream(self):
        selected = self.tree.selection()
        if not selected:
            return None
        item = selected[0]
        return self.stream_data_dict.get(item)

    def add_stream(self):
        dialog = StreamConfigDialog(self.window)
        self.window.wait_window(dialog)
        if dialog.result:
            used_cells = [s.get('cell_index', -1) for s in self.streams]
            cell_index = 0
            while cell_index in used_cells and cell_index < 20:
                cell_index += 1
            if cell_index >= 20:
                messagebox.showerror("Ошибка", "Достигнут лимит в 20 потоков")
                return
            existing_ids = [s['id'] for s in self.streams]
            stream_id = generate_stream_id(existing_ids)
            new_stream = {
                'id': stream_id,
                'name': dialog.result['name'],
                'url': dialog.result['url'],
                'fps': dialog.result['fps'],
                'ip': dialog.result['ip'],
                'port': dialog.result['port'],
                'username': dialog.result['username'],
                'password': dialog.result['password'],
                'path': dialog.result['path'],
                'detection_type': dialog.result['detection_type'],
                'detection_level': dialog.result['detection_level'],
                'cell_index': cell_index,
                'triggers': dialog.result['triggers'],
                'connection_type': dialog.result.get('connection_type', 'rtsp')
            }
            self.streams.append(new_stream)
            self.load_streams()

    def edit_stream(self):
        stream = self.get_selected_stream()
        if not stream:
            messagebox.showwarning("Внимание", "Выберите поток для редактирования")
            return
        if stream['id'] in self.parent.stream_threads:
            self.parent.stop_stream(stream['id'])
        dialog = StreamConfigDialog(self.window, {
            'name': stream['name'],
            'ip': stream.get('ip', '192.168.1.100'),
            'port': stream.get('port', '554'),
            'username': stream.get('username', 'admin'),
            'password': stream.get('password', '123456'),
            'path': stream.get('path', '/stream'),
            'fps': stream.get('fps', 15),
            'detection_type': stream.get('detection_type', 'yolo'),
            'detection_level': stream.get('detection_level', 'medium'),
            'triggers': stream.get('triggers', {
                'person': True,
                'vehicle': True,
                'motion': True
            }),
            'connection_type': stream.get('connection_type', 'rtsp')
        })
        self.window.wait_window(dialog)
        if dialog.result:
            for i, s in enumerate(self.streams):
                if s['id'] == stream['id']:
                    self.streams[i] = {
                        'id': stream['id'],
                        'name': dialog.result['name'],
                        'url': dialog.result['url'],
                        'fps': dialog.result['fps'],
                        'ip': dialog.result['ip'],
                        'port': dialog.result['port'],
                        'username': dialog.result['username'],
                        'password': dialog.result['password'],
                        'path': dialog.result['path'],
                        'detection_type': dialog.result['detection_type'],
                        'detection_level': dialog.result['detection_level'],
                        'cell_index': stream.get('cell_index', 0),
                        'triggers': dialog.result['triggers'],
                        'connection_type': dialog.result.get('connection_type', 'rtsp')
                    }
                    break
            self.load_streams()

    def delete_stream(self):
        stream = self.get_selected_stream()
        if not stream:
            messagebox.showwarning("Внимание", "Выберите поток для удаления")
            return
        if messagebox.askyesno("Подтверждение", f"Удалить поток '{stream['name']}'?"):
            if stream['id'] in self.parent.stream_threads:
                self.parent.stop_stream(stream['id'])
            self.streams = [s for s in self.streams if s['id'] != stream['id']]
            self.load_streams()
            messagebox.showinfo("Успех", f"Поток '{stream['name']}' удален")

    def on_double_click(self, event):
        stream = self.get_selected_stream()
        if stream and hasattr(self.parent, 'open_fullscreen_from_selector'):
            self.parent.open_fullscreen_from_selector(stream)

    def on_ok(self):
        self.result = self.streams
        self.window.grab_release()
        self.window.destroy()

    def on_save(self):
        if self.on_stream_selected:
            for stream in self.streams:
                self.on_stream_selected(stream)
        messagebox.showinfo("Сохранено", "Изменения сохранены")
        self.result = self.streams

class StreamConfigDialog(tk.Toplevel):
    def __init__(self, parent, stream_data=None):
        super().__init__(parent)
        self.parent = parent
        self.title("Настройка RTSP потока")
        self.geometry("450x600")
        self.resizable(False, False)
        self.bg_color = '#f0f2f5'
        self.fg_color = '#1a1a1a'
        self.accent_color = '#3b82f6'
        self.success_color = '#10b981'
        self.configure(bg=self.bg_color)
        self.stream_data = stream_data or {}
        self.result = None
        self.setup_ui()
        self.center_window()
        self.generate_url()
        self.on_connection_type_change()
        self.grab_set()

    def center_window(self):
        self.update_idletasks()
        width = self.winfo_width()
        height = self.winfo_height()
        x = (self.winfo_screenwidth() // 2) - (width // 2)
        y = (self.winfo_screenheight() // 2) - (height // 2)
        self.geometry(f'{width}x{height}+{x}+{y}')

    def setup_ui(self):
        main_frame = tk.Frame(self, bg=self.bg_color)
        main_frame.pack(fill=tk.BOTH, expand=True, padx=15, pady=15)
        title_label = tk.Label(main_frame, text="Настройка RTSP потока",
                                font=("Segoe UI", 14, "bold"),
                                bg=self.bg_color, fg=self.accent_color)
        title_label.grid(row=0, column=0, columnspan=2, pady=(0, 15), sticky=tk.W)
        tk.Label(main_frame, text="Тип подключения:",
                bg=self.bg_color, fg=self.fg_color,
                font=('Segoe UI', 9)).grid(row=1, column=0, sticky=tk.W, pady=5)
        self.connection_type_var = tk.StringVar(value=self.stream_data.get('connection_type', 'rtsp'))
        connection_types = ['rtsp']
        if HAS_ONVIF:
            connection_types.append('onvif')
        connection_combo = ttk.Combobox(main_frame, textvariable=self.connection_type_var,
                                       values=connection_types,
                                       width=18, state='readonly', font=('Segoe UI', 9))
        connection_combo.grid(row=1, column=1, sticky=tk.W, pady=5, padx=(10, 0))
        connection_combo.bind('<<ComboboxSelected>>', self.on_connection_type_change)
        tk.Label(main_frame, text="Имя камеры:",
                bg=self.bg_color, fg=self.fg_color,
                font=('Segoe UI', 9)).grid(row=2, column=0, sticky=tk.W, pady=5)
        self.name_var = tk.StringVar(value=self.stream_data.get('name', f"Камера {datetime.now().strftime('%H%M%S')}"))
        self.name_entry = tk.Entry(main_frame, textvariable=self.name_var,
                                   width=25, font=('Segoe UI', 9), bg='white', fg=self.fg_color,
                                   relief=tk.SOLID, borderwidth=1)
        self.name_entry.grid(row=2, column=1, pady=5, padx=(10, 0), sticky=tk.W)
        tk.Label(main_frame, text="IP адрес:",
                bg=self.bg_color, fg=self.fg_color,
                font=('Segoe UI', 9)).grid(row=3, column=0, sticky=tk.W, pady=5)
        self.ip_var = tk.StringVar(value=self.stream_data.get('ip', '192.168.1.100'))
        self.ip_entry = tk.Entry(main_frame, textvariable=self.ip_var,
                                width=25, font=('Segoe UI', 9), bg='white', fg=self.fg_color,
                                relief=tk.SOLID, borderwidth=1)
        self.ip_entry.grid(row=3, column=1, pady=5, padx=(10, 0), sticky=tk.W)
        tk.Label(main_frame, text="Порт:",
                bg=self.bg_color, fg=self.fg_color,
                font=('Segoe UI', 9)).grid(row=4, column=0, sticky=tk.W, pady=5)
        self.port_var = tk.StringVar(value=self.stream_data.get('port', '554'))
        self.port_entry = tk.Entry(main_frame, textvariable=self.port_var,
                                  width=10, font=('Segoe UI', 9), bg='white', fg=self.fg_color,
                                  relief=tk.SOLID, borderwidth=1)
        self.port_entry.grid(row=4, column=1, sticky=tk.W, pady=5, padx=(10, 0))
        tk.Label(main_frame, text="Имя пользователя:",
                bg=self.bg_color, fg=self.fg_color,
                font=('Segoe UI', 9)).grid(row=5, column=0, sticky=tk.W, pady=5)
        self.username_var = tk.StringVar(value=self.stream_data.get('username', 'admin'))
        self.username_entry = tk.Entry(main_frame, textvariable=self.username_var,
                                      width=20, font=('Segoe UI', 9), bg='white', fg=self.fg_color,
                                      relief=tk.SOLID, borderwidth=1)
        self.username_entry.grid(row=5, column=1, pady=5, padx=(10, 0), sticky=tk.W)
        tk.Label(main_frame, text="Пароль:",
                bg=self.bg_color, fg=self.fg_color,
                font=('Segoe UI', 9)).grid(row=6, column=0, sticky=tk.W, pady=5)
        self.password_var = tk.StringVar(value=self.stream_data.get('password', '123456'))
        self.password_entry = tk.Entry(main_frame, textvariable=self.password_var,
                                      width=20, show="•", font=('Segoe UI', 9),
                                      bg='white', fg=self.fg_color,
                                      relief=tk.SOLID, borderwidth=1)
        self.password_entry.grid(row=6, column=1, pady=5, padx=(10, 0), sticky=tk.W)
        tk.Label(main_frame, text="Путь к потоку:",
                bg=self.bg_color, fg=self.fg_color,
                font=('Segoe UI', 9)).grid(row=7, column=0, sticky=tk.W, pady=5)
        self.path_var = tk.StringVar(value=self.stream_data.get('path', '/stream'))
        self.path_entry = tk.Entry(main_frame, textvariable=self.path_var,
                                  width=25, font=('Segoe UI', 9), bg='white', fg=self.fg_color,
                                  relief=tk.SOLID, borderwidth=1)
        self.path_entry.grid(row=7, column=1, pady=5, padx=(10, 0), sticky=tk.W)
        tk.Label(main_frame, text="FPS обработки:",
                bg=self.bg_color, fg=self.fg_color,
                font=('Segoe UI', 9)).grid(row=8, column=0, sticky=tk.W, pady=5)
        self.fps_var = tk.IntVar(value=self.stream_data.get('fps', 15))
        self.fps_spinbox = tk.Spinbox(main_frame, from_=1, to=30,
                                      textvariable=self.fps_var, width=10,
                                      font=('Segoe UI', 9), bg='white', fg=self.fg_color)
        self.fps_spinbox.grid(row=8, column=1, sticky=tk.W, pady=5, padx=(10, 0))
        tk.Label(main_frame, text="Тип детекции:",
                bg=self.bg_color, fg=self.fg_color,
                font=('Segoe UI', 9)).grid(row=9, column=0, sticky=tk.W, pady=5)
        self.detection_type_var = tk.StringVar(value=self.stream_data.get('detection_type', 'yolo'))
        detection_combo = ttk.Combobox(main_frame, textvariable=self.detection_type_var,
                                      values=['yolo', 'motion_only'],
                                      width=18, state='readonly', font=('Segoe UI', 9))
        detection_combo.grid(row=9, column=1, sticky=tk.W, pady=5, padx=(10, 0))
        tk.Label(main_frame, text="Уровень детекции:",
                bg=self.bg_color, fg=self.fg_color,
                font=('Segoe UI', 9)).grid(row=10, column=0, sticky=tk.W, pady=5)
        self.detection_level_var = tk.StringVar(value=self.stream_data.get('detection_level', 'medium'))
        level_combo = ttk.Combobox(main_frame, textvariable=self.detection_level_var,
                                  values=['low', 'medium', 'high', 'maximum'],
                                  width=18, state='readonly', font=('Segoe UI', 9))
        level_combo.grid(row=10, column=1, sticky=tk.W, pady=5, padx=(10, 0))
        tk.Label(main_frame, text="Триггеры детекции:",
                bg=self.bg_color, fg=self.fg_color,
                font=('Segoe UI', 9)).grid(row=11, column=0, sticky=tk.W, pady=5)
        triggers_frame = tk.Frame(main_frame, bg=self.bg_color)
        triggers_frame.grid(row=11, column=1, sticky=tk.W, pady=5, padx=(10, 0))
        self.trigger_person_var = tk.BooleanVar(value=self.stream_data.get('triggers', {}).get('person', True))
        self.trigger_vehicle_var = tk.BooleanVar(value=self.stream_data.get('triggers', {}).get('vehicle', True))
        self.trigger_motion_var = tk.BooleanVar(value=self.stream_data.get('triggers', {}).get('motion', True))
        ttk.Checkbutton(triggers_frame, text="👤 Человек", variable=self.trigger_person_var).pack(side=tk.LEFT, padx=(0, 10))
        ttk.Checkbutton(triggers_frame, text="🚗 Автомобиль", variable=self.trigger_vehicle_var).pack(side=tk.LEFT, padx=(0, 10))
        ttk.Checkbutton(triggers_frame, text="🎥 Движение", variable=self.trigger_motion_var).pack(side=tk.LEFT)
        tk.Label(main_frame, text="Сгенерированный URL:",
                bg=self.bg_color, fg=self.fg_color,
                font=('Segoe UI', 9)).grid(row=12, column=0, sticky=tk.W, pady=5)
        url_frame = tk.Frame(main_frame, bg=self.bg_color)
        url_frame.grid(row=12, column=1, sticky=tk.W, pady=5, padx=(10, 0))
        self.url_var = tk.StringVar()
        self.url_display = tk.Label(url_frame, textvariable=self.url_var,
                                    relief=tk.SUNKEN, padx=5, pady=5,
                                    bg="white", fg=self.accent_color,
                                    font=('Consolas', 8), width=35, wraplength=300,
                                    anchor=tk.W, justify=tk.LEFT)
        self.url_display.pack()
        button_frame = tk.Frame(main_frame, bg=self.bg_color)
        button_frame.grid(row=13, column=0, columnspan=2, pady=15)
        button_style = {
            'bg': self.accent_color,
            'fg': 'white',
            'font': ('Segoe UI', 9, 'bold'),
            'padx': 8,
            'pady': 4,
            'relief': tk.FLAT,
            'borderwidth': 0,
            'cursor': 'hand2'
        }
        if HAS_ONVIF:
            tk.Button(button_frame, text="🌐 Onvif Scan",
                     command=self.scan_onvif, **button_style).pack(side=tk.LEFT, padx=2)
        tk.Button(button_frame, text="🔧 Тест",
                 command=self.test_connection, **button_style).pack(side=tk.LEFT, padx=2)
        tk.Button(button_frame, text="✅ OK",
                 command=self.on_ok,
                 bg=self.success_color, fg='white',
                 font=('Segoe UI', 9, 'bold'),
                 padx=8, pady=4,
                 relief=tk.FLAT).pack(side=tk.LEFT, padx=2)
        tk.Button(button_frame, text="❌ Отмена",
                 command=self.destroy,
                 bg='#ef4444', fg='white',
                 font=('Segoe UI', 9, 'bold'),
                 padx=8, pady=4,
                 relief=tk.FLAT).pack(side=tk.LEFT, padx=2)
        entries = [self.name_entry, self.ip_entry, self.port_entry,
                  self.username_entry, self.password_entry, self.path_entry]
        for entry in entries:
            entry.bind('<KeyRelease>', lambda e: self.generate_url())

    def on_connection_type_change(self, event=None):
        if self.connection_type_var.get() == 'onvif':
            self.path_entry.config(state='disabled')
        else:
            self.path_entry.config(state='normal')
        self.generate_url()

    def generate_url(self):
        try:
            connection_type = self.connection_type_var.get()
            ip = self.ip_var.get().strip()
            port = self.port_var.get().strip()
            username = self.username_var.get().strip()
            password = self.password_var.get().strip()
            if connection_type == 'rtsp':
                path = self.path_var.get().strip()
                if path and not path.startswith('/'):
                    path = '/' + path
                if username and password:
                    url = f"rtsp://{username}:{password}@{ip}:{port}{path}"
                else:
                    url = f"rtsp://{ip}:{port}{path}"
            else:
                url = f"onvif://{username}:{password}@{ip}:{port}"
            self.url_var.set(url)
            return url
        except Exception as e:
            self.url_var.set(f"Ошибка: {str(e)}")
            return None

    def scan_onvif(self):
        if not HAS_ONVIF:
            messagebox.showwarning("ONVIF не доступен",
                                 "Библиотека ONVIF не установлена.\nУстановите: pip install onvif-zeep")
            return
        ip = self.ip_var.get().strip()
        if not validate_ip(ip):
            messagebox.showerror("Ошибка", "Неверный IP адрес")
            return
        port = self.port_var.get().strip()
        if not validate_port(port):
            messagebox.showerror("Ошибка", "Неверный порт (допустимо 1-65535)")
            return
        username = self.username_var.get().strip()
        password = self.password_var.get().strip()
        try:
            with SuppressOutput():
                camera = ONVIFCamera(ip, int(port), username, password)
            device_info = camera.devicemgmt.GetDeviceInformation()
            media_service = camera.create_media_service()
            profiles = media_service.GetProfiles()
            info_text = f"Устройство: {device_info.Manufacturer} {device_info.Model}\n"
            info_text += f"Серийный номер: {device_info.SerialNumber}\n"
            info_text += f"Профилей: {len(profiles)}\n\n"
            stream_urls = []
            for i, profile in enumerate(profiles):
                try:
                    stream_uri = media_service.GetStreamUri({
                        'StreamSetup': {
                            'Stream': 'RTP-Unicast',
                            'Transport': {'Protocol': 'RTSP'}
                        },
                        'ProfileToken': profile.token
                    })
                    rtsp_url = stream_uri.Uri
                    if username and password:
                        rtsp_url = rtsp_url.replace('rtsp://', f'rtsp://{username}:{password}@')
                    stream_urls.append(rtsp_url)
                    info_text += f"Профиль {i+1}: {rtsp_url}\n"
                except Exception as e:
                    info_text += f"Профиль {i+1}: Ошибка получения URL\n"
            if stream_urls:
                self.path_var.set('')
                self.connection_type_var.set('rtsp')
                self.url_var.set(stream_urls[0])
            messagebox.showinfo("Onvif Scan", info_text)
        except Exception as e:
            messagebox.showerror("Ошибка", f"Не удалось подключиться к Onvif устройству: {str(e)}")

    def test_connection(self):
        url = self.generate_url()
        if not url:
            messagebox.showerror("Ошибка", "Не удалось сгенерировать URL")
            return
        ip = self.ip_var.get().strip()
        if not validate_ip(ip):
            messagebox.showerror("Ошибка", "Неверный IP адрес")
            return
        if not validate_port(self.port_var.get()):
            messagebox.showerror("Ошибка", "Неверный порт (1-65535)")
            return
        try:
            if url.startswith('onvif://'):
                import subprocess
                try:
                    result = subprocess.run(['ping', '-n', '1', '-w', '1000', ip],
                                          capture_output=True, text=True)
                    if result.returncode == 0:
                        messagebox.showinfo("Успех", f"Камера доступна по IP: {ip}")
                    else:
                        messagebox.showerror("Ошибка", f"Камера недоступна по IP: {ip}")
                except:
                    messagebox.showerror("Ошибка", "Не удалось выполнить ping")
                return
            cap = cv2.VideoCapture(url)
            if not cap.isOpened():
                messagebox.showerror("Ошибка", "Не удалось подключиться к камере")
                return
            ret, frame = cap.read()
            cap.release()
            if ret and frame is not None:
                cv2.imshow(f"Тест: {self.name_var.get()}", frame)
                cv2.waitKey(2000)
                cv2.destroyWindow(f"Тест: {self.name_var.get()}")
                messagebox.showinfo("Успех", "Подключение к камере успешно установлено!")
            else:
                messagebox.showwarning("Предупреждение",
                    "Подключение установлено, но не удалось получить кадр")
        except Exception as e:
            messagebox.showerror("Ошибка", f"Ошибка подключения: {str(e)}")

    def on_ok(self):
        if not self.name_var.get().strip():
            messagebox.showerror("Ошибка", "Введите имя камеры")
            return
        ip = self.ip_var.get().strip()
        if not validate_ip(ip):
            messagebox.showerror("Ошибка", "Введите корректный IP адрес")
            return
        if not validate_port(self.port_var.get()):
            messagebox.showerror("Ошибка", "Введите корректный порт (1-65535)")
            return
        if not (self.trigger_person_var.get() or self.trigger_vehicle_var.get() or self.trigger_motion_var.get()):
            messagebox.showerror("Ошибка", "Выберите хотя бы один триггер для детекции")
            return
        url = self.generate_url()
        if not url:
            messagebox.showerror("Ошибка", "Не удалось сгенерировать URL")
            return
        self.result = {
            'name': self.name_var.get().strip(),
            'ip': ip,
            'port': self.port_var.get().strip(),
            'username': self.username_var.get().strip(),
            'password': self.password_var.get().strip(),
            'path': self.path_var.get().strip(),
            'fps': self.fps_var.get(),
            'detection_type': self.detection_type_var.get(),
            'detection_level': self.detection_level_var.get(),
            'triggers': {
                'person': self.trigger_person_var.get(),
                'vehicle': self.trigger_vehicle_var.get(),
                'motion': self.trigger_motion_var.get()
            },
            'url': url,
            'connection_type': self.connection_type_var.get()
        }
        self.grab_release()
        self.destroy()

class SettingsWindow:
    def __init__(self, parent, config, on_save_callback, model_manager=None):
        self.parent = parent
        self.config = config
        self.on_save_callback = on_save_callback
        self.model_manager = model_manager
        self.window = tk.Toplevel(parent)
        self.window.title("Настройки")
        self.window.geometry("500x650")
        self.window.resizable(False, False)
        self.bg_color = '#f0f2f5'
        self.panel_bg = '#ffffff'
        self.fg_color = '#1a1a1a'
        self.accent_color = '#3b82f6'
        self.success_color = '#10b981'
        self.window.configure(bg=self.bg_color)
        self.center_window()
        self.setup_ui()
        self.window.grab_set()

    def center_window(self):
        self.window.update_idletasks()
        width = self.window.winfo_width()
        height = self.window.winfo_height()
        x = (self.window.winfo_screenwidth() // 2) - (width // 2)
        y = (self.window.winfo_screenheight() // 2) - (height // 2)
        self.window.geometry(f'{width}x{height}+{x}+{y}')

    def setup_ui(self):
        notebook = ttk.Notebook(self.window)
        notebook.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        perf_frame = tk.Frame(notebook, bg=self.panel_bg)
        notebook.add(perf_frame, text="⚡ Производительность")
        self.setup_performance_tab(perf_frame)
        detect_frame = tk.Frame(notebook, bg=self.panel_bg)
        notebook.add(detect_frame, text="🎯 Детекция")
        self.setup_detection_tab(detect_frame)
        save_frame = tk.Frame(notebook, bg=self.panel_bg)
        notebook.add(save_frame, text="💾 Сохранение")
        self.setup_save_tab(save_frame)
        materials_frame = tk.Frame(notebook, bg=self.panel_bg)
        notebook.add(materials_frame, text="🏗️ Материалы")
        self.setup_materials_tab(materials_frame)
        anpr_frame = tk.Frame(notebook, bg=self.panel_bg)
        notebook.add(anpr_frame, text="🚗 ANPR")
        self.setup_anpr_tab(anpr_frame)
        button_frame = tk.Frame(self.window, bg=self.bg_color)
        button_frame.pack(fill=tk.X, padx=10, pady=(0, 10))
        tk.Button(button_frame, text="✅ Применить",
                 command=self.on_apply,
                 bg=self.success_color, fg='white',
                 font=('Segoe UI', 10, 'bold'),
                 padx=20, pady=8).pack(side=tk.RIGHT, padx=5)
        tk.Button(button_frame, text="❌ Отмена",
                 command=self.window.destroy,
                 bg='#ef4444', fg='white',
                 font=('Segoe UI', 10, 'bold'),
                 padx=20, pady=8).pack(side=tk.RIGHT, padx=5)

    def setup_performance_tab(self, parent):
        parent.configure(bg=self.panel_bg)
        ui_frame = tk.Frame(parent, bg=self.panel_bg)
        ui_frame.pack(fill=tk.X, padx=20, pady=(20, 10))
        tk.Label(ui_frame, text="Интервал обновления UI (мс):",
                bg=self.panel_bg, fg=self.fg_color,
                font=('Segoe UI', 10)).pack(side=tk.LEFT)
        self.ui_update_var = tk.IntVar(value=self.config.config['performance'].get('ui_update_interval', 33))
        ui_spinbox = tk.Spinbox(ui_frame, from_=16, to=200,
                               textvariable=self.ui_update_var,
                               width=10,
                               font=('Segoe UI', 10))
        ui_spinbox.pack(side=tk.RIGHT)
        skip_frame = tk.Frame(parent, bg=self.panel_bg)
        skip_frame.pack(fill=tk.X, padx=20, pady=10)
        tk.Label(skip_frame, text="Пропуск кадров (1-30):",
                bg=self.panel_bg, fg=self.fg_color,
                font=('Segoe UI', 10)).pack(side=tk.LEFT)
        self.skip_var = tk.IntVar(value=self.config.config['performance'].get('frame_skip_factor', 1))
        skip_spinbox = tk.Spinbox(skip_frame, from_=1, to=30,
                                 textvariable=self.skip_var,
                                 width=10,
                                 font=('Segoe UI', 10))
        skip_spinbox.pack(side=tk.RIGHT)
        parallel_frame = tk.Frame(parent, bg=self.panel_bg)
        parallel_frame.pack(fill=tk.X, padx=20, pady=10)
        self.parallel_var = tk.BooleanVar(value=self.config.config['performance'].get('parallel_processing', True))
        ttk.Checkbutton(parallel_frame, text="Параллельная обработка",
                       variable=self.parallel_var).pack(anchor=tk.W)
        cpu_frame = tk.Frame(parent, bg=self.panel_bg)
        cpu_frame.pack(fill=tk.X, padx=20, pady=10)
        tk.Label(cpu_frame, text="Рабочие потоки CPU:",
                bg=self.panel_bg, fg=self.fg_color,
                font=('Segoe UI', 10)).pack(side=tk.LEFT)
        self.cpu_workers_var = tk.IntVar(value=self.config.config['performance'].get('cpu_workers', 1))
        cpu_spinbox = tk.Spinbox(cpu_frame, from_=1, to=8,
                                textvariable=self.cpu_workers_var,
                                width=10,
                                font=('Segoe UI', 10))
        cpu_spinbox.pack(side=tk.RIGHT)

    def setup_detection_tab(self, parent):
        parent.configure(bg=self.panel_bg)
        conf_frame = tk.Frame(parent, bg=self.panel_bg)
        conf_frame.pack(fill=tk.X, padx=20, pady=(20, 10))
        tk.Label(conf_frame, text="Порог уверенности:",
                bg=self.panel_bg, fg=self.fg_color,
                font=('Segoe UI', 10)).pack(side=tk.LEFT)
        self.confidence_var = tk.DoubleVar(value=self.config.config['detection'].get('confidence_threshold', 0.5))
        conf_spinbox = tk.Spinbox(conf_frame, from_=0.1, to=1.0, increment=0.1,
                                 textvariable=self.confidence_var,
                                 width=10,
                                 font=('Segoe UI', 10))
        conf_spinbox.pack(side=tk.RIGHT)
        track_frame = tk.Frame(parent, bg=self.panel_bg)
        track_frame.pack(fill=tk.X, padx=20, pady=10)
        self.track_var = tk.BooleanVar(value=self.config.config['detection'].get('track_objects', True))
        ttk.Checkbutton(track_frame, text="Отслеживание объектов",
                       variable=self.track_var).pack(anchor=tk.W)
        type_frame = tk.Frame(parent, bg=self.panel_bg)
        type_frame.pack(fill=tk.X, padx=20, pady=10)
        tk.Label(type_frame, text="Тип детекции:",
                bg=self.panel_bg, fg=self.fg_color,
                font=('Segoe UI', 10)).pack(side=tk.LEFT)
        self.detection_type_var = tk.StringVar(value=self.config.config['detection'].get('detection_type', 'yolo'))
        type_combo = ttk.Combobox(type_frame, textvariable=self.detection_type_var,
                                 values=['yolo', 'motion_only'],
                                 width=15, state='readonly', font=('Segoe UI', 10))
        type_combo.pack(side=tk.RIGHT)
        dir_frame = tk.Frame(parent, bg=self.panel_bg)
        dir_frame.pack(fill=tk.X, padx=20, pady=10)
        tk.Label(dir_frame, text="Папка с моделями YOLOv26:",
                bg=self.panel_bg, fg=self.fg_color,
                font=('Segoe UI', 10)).pack(anchor=tk.W)
        self.yolo_dir_var = tk.StringVar(value=self.config.config['detection'].get('yolo_models_dir', 'models'))
        dir_entry = tk.Entry(dir_frame, textvariable=self.yolo_dir_var,
                           font=('Segoe UI', 10), width=40)
        dir_entry.pack(fill=tk.X, pady=(5, 0))
        tk.Button(dir_frame, text="📁 Выбрать",
                 command=self.browse_yolo_dir,
                 bg=self.accent_color, fg='white',
                 font=('Segoe UI', 9, 'bold'),
                 padx=10, pady=4).pack(pady=(5, 0))
        model_frame = tk.Frame(parent, bg=self.panel_bg)
        model_frame.pack(fill=tk.X, padx=20, pady=10)
        tk.Label(model_frame, text="Модель YOLOv26:",
                bg=self.panel_bg, fg=self.fg_color,
                font=('Segoe UI', 10)).pack(side=tk.LEFT)
        self.refresh_available_models()
        self.yolo_model_var = tk.StringVar(value=self.config.config['detection'].get('yolo_model', 'yolo26n.pt'))
        self.yolo_model_combo = ttk.Combobox(model_frame, textvariable=self.yolo_model_var,
                                            values=[model['name'] for model in self.available_models],
                                            width=30, state='readonly', font=('Segoe UI', 10))
        self.yolo_model_combo.pack(side=tk.RIGHT)
        dir_entry.bind('<KeyRelease>', lambda e: self.refresh_available_models())

    def setup_materials_tab(self, parent):
        parent.configure(bg=self.panel_bg)
        enabled_frame = tk.Frame(parent, bg=self.panel_bg)
        enabled_frame.pack(fill=tk.X, padx=20, pady=(20, 10))
        self.materials_enabled_var = tk.BooleanVar(value=self.config.config['materials'].get('enabled', True))
        ttk.Checkbutton(enabled_frame, text="Включить распознавание нерудных материалов",
                       variable=self.materials_enabled_var).pack(anchor=tk.W)
        
        yolo_frame = tk.Frame(parent, bg=self.panel_bg)
        yolo_frame.pack(fill=tk.X, padx=20, pady=10)
        self.materials_yolo_var = tk.BooleanVar(value=self.config.config['materials'].get('use_yolo', True))
        ttk.Checkbutton(yolo_frame, text="Использовать YOLO v26 модель для распознавания материалов",
                       variable=self.materials_yolo_var).pack(anchor=tk.W)
        
        threshold_frame = tk.Frame(parent, bg=self.panel_bg)
        threshold_frame.pack(fill=tk.X, padx=20, pady=10)
        tk.Label(threshold_frame, text="Порог уверенности распознавания:",
                bg=self.panel_bg, fg=self.fg_color,
                font=('Segoe UI', 10)).pack(side=tk.LEFT)
        self.materials_threshold_var = tk.DoubleVar(value=self.config.config['materials'].get('detection_threshold', 0.5))
        threshold_spinbox = tk.Spinbox(threshold_frame, from_=0.1, to=1.0, increment=0.1,
                                      textvariable=self.materials_threshold_var,
                                      width=10,
                                      font=('Segoe UI', 10))
        threshold_spinbox.pack(side=tk.RIGHT)
        
        model_path_frame = tk.Frame(parent, bg=self.panel_bg)
        model_path_frame.pack(fill=tk.X, padx=20, pady=10)
        tk.Label(model_path_frame, text="Путь к модели YOLO для материалов:",
                bg=self.panel_bg, fg=self.fg_color,
                font=('Segoe UI', 10)).pack(anchor=tk.W)
        self.materials_yolo_path_var = tk.StringVar(value=self.config.config['materials'].get('yolo_model_path', 'models/materials/materials.pt'))
        yolo_path_entry = tk.Entry(model_path_frame, textvariable=self.materials_yolo_path_var,
                                 font=('Segoe UI', 10), width=40)
        yolo_path_entry.pack(fill=tk.X, pady=(5, 0))
        tk.Button(model_path_frame, text="📁 Выбрать",
                 command=lambda: self.browse_file(self.materials_yolo_path_var, [("YOLO модели", "*.pt")]),
                 bg=self.accent_color, fg='white',
                 font=('Segoe UI', 9, 'bold'),
                 padx=10, pady=4).pack(pady=(5, 0))
        
        list_frame = tk.Frame(parent, bg=self.panel_bg)
        list_frame.pack(fill=tk.BOTH, expand=True, padx=20, pady=10)
        tk.Label(list_frame, text="Список материалов для распознавания:",
                bg=self.panel_bg, fg=self.fg_color,
                font=('Segoe UI', 10)).pack(anchor=tk.W)
        materials_list = tk.Frame(list_frame, bg=self.panel_bg)
        materials_list.pack(fill=tk.BOTH, expand=True, pady=(5, 0))
        self.materials_listbox = tk.Listbox(materials_list, height=8,
                                           font=('Segoe UI', 10),
                                           selectmode=tk.SINGLE)
        scrollbar = tk.Scrollbar(materials_list)
        self.materials_listbox.config(yscrollcommand=scrollbar.set)
        scrollbar.config(command=self.materials_listbox.yview)
        self.materials_listbox.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        self.refresh_materials_list()
        controls_frame = tk.Frame(list_frame, bg=self.panel_bg)
        controls_frame.pack(fill=tk.X, pady=(5, 0))
        tk.Button(controls_frame, text="➕ Добавить",
                 command=self.add_material,
                 bg=self.success_color, fg='white',
                 font=('Segoe UI', 9, 'bold'),
                 padx=8, pady=4).pack(side=tk.LEFT, padx=2)
        tk.Button(controls_frame, text="✏️ Редактировать",
                 command=self.edit_material,
                 bg=self.accent_color, fg='white',
                 font=('Segoe UI', 9, 'bold'),
                 padx=8, pady=4).pack(side=tk.LEFT, padx=2)
        tk.Button(controls_frame, text="🗑️ Удалить",
                 command=self.remove_material,
                 bg='#ef4444', fg='white',
                 font=('Segoe UI', 9, 'bold'),
                 padx=8, pady=4).pack(side=tk.LEFT, padx=2)

    def refresh_materials_list(self):
        self.materials_listbox.delete(0, tk.END)
        materials = self.config.get_materials()
        for material in materials:
            self.materials_listbox.insert(tk.END, material)

    def add_material(self):
        material_name = simpledialog.askstring("Добавить материал", "Введите название материала:")
        if material_name:
            if self.config.add_material(material_name):
                self.refresh_materials_list()
                messagebox.showinfo("Успех", f"Материал '{material_name}' добавлен")
            else:
                messagebox.showwarning("Предупреждение", f"Материал '{material_name}' уже существует")

    def edit_material(self):
        selection = self.materials_listbox.curselection()
        if not selection:
            messagebox.showwarning("Предупреждение", "Выберите материал для редактирования")
            return
        index = selection[0]
        old_name = self.materials_listbox.get(index)
        new_name = simpledialog.askstring("Редактировать материал", "Введите новое название:", initialvalue=old_name)
        if new_name and new_name != old_name:
            if self.config.remove_material(old_name):
                if self.config.add_material(new_name):
                    self.refresh_materials_list()
                    messagebox.showinfo("Успех", f"Материал '{old_name}' изменен на '{new_name}'")
                else:
                    self.config.add_material(old_name)
                    messagebox.showwarning("Ошибка", "Не удалось изменить материал")
            else:
                messagebox.showwarning("Ошибка", "Не удалось удалить старый материал")

    def remove_material(self):
        selection = self.materials_listbox.curselection()
        if not selection:
            messagebox.showwarning("Предупреждение", "Выберите материал для удаления")
            return
        index = selection[0]
        material_name = self.materials_listbox.get(index)
        if messagebox.askyesno("Подтверждение", f"Удалить материал '{material_name}'?"):
            if self.config.remove_material(material_name):
                self.refresh_materials_list()
                messagebox.showinfo("Успех", f"Материал '{material_name}' удален")
            else:
                messagebox.showwarning("Ошибка", "Не удалось удалить материал")

    def setup_save_tab(self, parent):
        parent.configure(bg=self.panel_bg)
        path_frame = tk.Frame(parent, bg=self.panel_bg)
        path_frame.pack(fill=tk.X, padx=20, pady=(20, 10))
        tk.Label(path_frame, text="Путь сохранения:",
                bg=self.panel_bg, fg=self.fg_color,
                font=('Segoe UI', 10)).pack(anchor=tk.W)
        self.save_path_var = tk.StringVar(value=self.config.config.get('save_path', 'saved_frames'))
        path_entry = tk.Entry(path_frame, textvariable=self.save_path_var,
                             font=('Segoe UI', 10), width=40)
        path_entry.pack(fill=tk.X, pady=(5, 0))
        tk.Button(path_frame, text="📁 Выбрать",
                 command=self.browse_save_path,
                 bg=self.accent_color, fg='white',
                 font=('Segoe UI', 9, 'bold'),
                 padx=10, pady=4).pack(pady=(5, 0))
        every_frame = tk.Frame(parent, bg=self.panel_bg)
        every_frame.pack(fill=tk.X, padx=20, pady=10)
        tk.Label(every_frame, text="Сохранять каждый N-ый кадр:",
                bg=self.panel_bg, fg=self.fg_color,
                font=('Segoe UI', 10)).pack(side=tk.LEFT)
        self.save_every_var = tk.IntVar(value=self.config.config['save_settings'].get('save_every_n_frame', 5))
        every_spinbox = tk.Spinbox(every_frame, from_=1, to=30,
                                  textvariable=self.save_every_var,
                                  width=10,
                                  font=('Segoe UI', 10))
        every_spinbox.pack(side=tk.RIGHT)
        max_frame = tk.Frame(parent, bg=self.panel_bg)
        max_frame.pack(fill=tk.X, padx=20, pady=10)
        tk.Label(max_frame, text="Макс. кадров на событие:",
                bg=self.panel_bg, fg=self.fg_color,
                font=('Segoe UI', 10)).pack(side=tk.LEFT)
        self.max_frames_var = tk.IntVar(value=self.config.config['save_settings'].get('max_frames_per_event', 50))
        max_spinbox = tk.Spinbox(max_frame, from_=1, to=500,
                                textvariable=self.max_frames_var,
                                width=10,
                                font=('Segoe UI', 10))
        max_spinbox.pack(side=tk.RIGHT)
        report_frame = tk.Frame(parent, bg=self.panel_bg)
        report_frame.pack(fill=tk.X, padx=20, pady=10)
        tk.Label(report_frame, text="Современный отчет событий:",
                bg=self.panel_bg, fg=self.fg_color,
                font=('Segoe UI', 10)).pack(anchor=tk.W)
        self.modern_report_var = tk.BooleanVar(value=True)
        ttk.Checkbutton(report_frame, text="Включить современный формат отчетов",
                       variable=self.modern_report_var).pack(anchor=tk.W, pady=(5, 0))
        tk.Label(report_frame, text="Формат: 'Время, Обнаружен транспорт N количество, у транспорта Обнаружен номер (номер)'",
                bg=self.panel_bg, fg='#666',
                font=('Segoe UI', 9)).pack(anchor=tk.W, pady=(2, 0))

    def setup_anpr_tab(self, parent):
        parent.configure(bg=self.panel_bg)
        enabled_frame = tk.Frame(parent, bg=self.panel_bg)
        enabled_frame.pack(fill=tk.X, padx=20, pady=(20, 10))
        self.anpr_enabled_var = tk.BooleanVar(value=self.config.config['anpr'].get('enabled', False))
        ttk.Checkbutton(enabled_frame, text="Включить систему распознавания номеров (ANPR)",
                       variable=self.anpr_enabled_var,
                       command=self.toggle_anpr_fields).pack(anchor=tk.W)
        yolo_frame = tk.Frame(parent, bg=self.panel_bg)
        yolo_frame.pack(fill=tk.X, padx=20, pady=10)
        tk.Label(yolo_frame, text="Путь к модели YOLO для номеров:",
                bg=self.panel_bg, fg=self.fg_color,
                font=('Segoe UI', 10)).pack(anchor=tk.W)
        self.anpr_yolo_var = tk.StringVar(value=self.config.config['anpr'].get('yolo_model_path', ANPRConfig.YOLO_MODEL_PATH))
        yolo_entry = tk.Entry(yolo_frame, textvariable=self.anpr_yolo_var,
                            font=('Segoe UI', 10), width=40,
                            state='disabled' if not self.anpr_enabled_var.get() else 'normal')
        yolo_entry.pack(fill=tk.X, pady=(5, 0))
        tk.Button(yolo_frame, text="📁 Выбрать",
                 command=lambda: self.browse_file(self.anpr_yolo_var, [("YOLO модели", "*.pt")]),
                 bg=self.accent_color, fg='white',
                 font=('Segoe UI', 9, 'bold'),
                 padx=10, pady=4,
                 state='disabled' if not self.anpr_enabled_var.get() else 'normal').pack(pady=(5, 0))
        ocr_frame = tk.Frame(parent, bg=self.panel_bg)
        ocr_frame.pack(fill=tk.X, padx=20, pady=10)
        tk.Label(ocr_frame, text="Путь к модели OCR для номеров:",
                bg=self.panel_bg, fg=self.fg_color,
                font=('Segoe UI', 10)).pack(anchor=tk.W)
        self.anpr_ocr_var = tk.StringVar(value=self.config.config['anpr'].get('ocr_model_path', ANPRConfig.OCR_MODEL_PATH))
        ocr_entry = tk.Entry(ocr_frame, textvariable=self.anpr_ocr_var,
                           font=('Segoe UI', 10), width=40,
                           state='disabled' if not self.anpr_enabled_var.get() else 'normal')
        ocr_entry.pack(fill=tk.X, pady=(5, 0))
        tk.Button(ocr_frame, text="📁 Выбрать",
                 command=lambda: self.browse_file(self.anpr_ocr_var, [("OCR модели", "*.pth")]),
                 bg=self.accent_color, fg='white',
                 font=('Segoe UI', 9, 'bold'),
                 padx=10, pady=4,
                 state='disabled' if not self.anpr_enabled_var.get() else 'normal').pack(pady=(5, 0))
        anpr_conf_frame = tk.Frame(parent, bg=self.panel_bg)
        anpr_conf_frame.pack(fill=tk.X, padx=20, pady=10)
        tk.Label(anpr_conf_frame, text="Порог уверенности ANPR:",
                bg=self.panel_bg, fg=self.fg_color,
                font=('Segoe UI', 10)).pack(side=tk.LEFT)
        self.anpr_confidence_var = tk.DoubleVar(value=self.config.config['anpr'].get('detection_confidence_threshold', 0.5))
        anpr_conf_spinbox = tk.Spinbox(anpr_conf_frame, from_=0.1, to=1.0, increment=0.1,
                                      textvariable=self.anpr_confidence_var,
                                      width=10,
                                      font=('Segoe UI', 10),
                                      state='disabled' if not self.anpr_enabled_var.get() else 'normal')
        anpr_conf_spinbox.pack(side=tk.RIGHT)
        separate_frame = tk.Frame(parent, bg=self.panel_bg)
        separate_frame.pack(fill=tk.X, padx=20, pady=10)
        self.anpr_separate_var = tk.BooleanVar(value=self.config.config['anpr'].get('use_separate_yolo', True))
        ttk.Checkbutton(separate_frame, text="Использовать отдельную модель YOLO для номеров",
                       variable=self.anpr_separate_var,
                       state='disabled' if not self.anpr_enabled_var.get() else 'normal').pack(anchor=tk.W)
        vehicles_frame = tk.Frame(parent, bg=self.panel_bg)
        vehicles_frame.pack(fill=tk.X, padx=20, pady=10)
        self.anpr_vehicles_var = tk.BooleanVar(value=self.config.config['anpr'].get('recognize_vehicles', True))
        ttk.Checkbutton(vehicles_frame, text="Распознавать номера у всех транспортных средств",
                       variable=self.anpr_vehicles_var,
                       state='disabled' if not self.anpr_enabled_var.get() else 'normal').pack(anchor=tk.W)

    def browse_yolo_dir(self):
        path = filedialog.askdirectory(
            title="Выберите папку с моделями YOLOv26",
            initialdir=self.yolo_dir_var.get()
        )
        if path:
            self.yolo_dir_var.set(path)
            self.refresh_available_models()

    def refresh_available_models(self):
        models_dir = self.yolo_dir_var.get()
        self.available_models = []
        yolo_model_patterns = ['yolo26n.pt', 'yolo26s.pt', 'yolo26m.pt', 'yolo26l.pt', 'yolo26x.pt']
        if os.path.exists(models_dir):
            for model_file in yolo_model_patterns:
                model_path = os.path.join(models_dir, model_file)
                if os.path.exists(model_path):
                    if 'yolo26n' in model_file:
                        display = f"YOLOv26 Nano (n) 🚀 Быстрее всего"
                    elif 'yolo26s' in model_file:
                        display = f"YOLOv26 Small (s) ⚡ Быстро"
                    elif 'yolo26m' in model_file:
                        display = f"YOLOv26 Medium (m) ⚖️ Сбалансированная"
                    elif 'yolo26l' in model_file:
                        display = f"YOLOv26 Large (l) 🎯 Точнее"
                    elif 'yolo26x' in model_file:
                        display = f"YOLOv26 XLarge (x) 🏆 Макс. точность"
                    else:
                        display = model_file
                    self.available_models.append({
                        'name': model_file,
                        'display': display,
                        'path': model_path,
                        'available': True
                    })
        if hasattr(self, 'yolo_model_combo'):
            self.yolo_model_combo['values'] = [model['name'] for model in self.available_models]
            current_model = self.yolo_model_var.get()
            if current_model not in [model['name'] for model in self.available_models] and self.available_models:
                self.yolo_model_var.set(self.available_models[0]['name'])

    def browse_save_path(self):
        path = filedialog.askdirectory(
            title="Выберите папку для сохранения",
            initialdir=self.save_path_var.get()
        )
        if path:
            self.save_path_var.set(path)

    def browse_file(self, var, filetypes):
        path = filedialog.askopenfilename(
            title="Выберите файл модели",
            filetypes=filetypes,
            initialdir=os.path.dirname(var.get()) if var.get() else "."
        )
        if path:
            var.set(path)

    def toggle_anpr_fields(self):
        state = 'normal' if self.anpr_enabled_var.get() else 'disabled'
        for widget in [self.anpr_yolo_var, self.anpr_ocr_var, self.anpr_confidence_var,
                      self.anpr_separate_var, self.anpr_vehicles_var]:
            if hasattr(widget, '_tk'):
                for w in widget._tk:
                    try:
                        w.configure(state=state)
                    except:
                        pass

    def on_apply(self):
        self.config.config['performance']['ui_update_interval'] = self.ui_update_var.get()
        self.config.config['performance']['frame_skip_factor'] = self.skip_var.get()
        self.config.config['performance']['parallel_processing'] = self.parallel_var.get()
        self.config.config['performance']['cpu_workers'] = self.cpu_workers_var.get()
        self.config.config['detection']['confidence_threshold'] = self.confidence_var.get()
        self.config.config['detection']['track_objects'] = self.track_var.get()
        self.config.config['detection']['detection_type'] = self.detection_type_var.get()
        self.config.config['detection']['yolo_models_dir'] = self.yolo_dir_var.get()
        self.config.config['detection']['yolo_model'] = self.yolo_model_var.get()
        selected_model = self.yolo_model_var.get()
        if 'yolo26n' in selected_model:
            model_type = 'yolo26n'
        elif 'yolo26s' in selected_model:
            model_type = 'yolo26s'
        elif 'yolo26m' in selected_model:
            model_type = 'yolo26m'
        elif 'yolo26l' in selected_model:
            model_type = 'yolo26l'
        elif 'yolo26x' in selected_model:
            model_type = 'yolo26x'
        else:
            model_type = 'yolo26n'
        self.config.config['detection']['yolo_model_type'] = model_type
        self.config.config['save_path'] = self.save_path_var.get()
        self.config.config['save_settings']['save_every_n_frame'] = self.save_every_var.get()
        self.config.config['save_settings']['max_frames_per_event'] = self.max_frames_var.get()
        self.config.config['materials']['enabled'] = self.materials_enabled_var.get()
        self.config.config['materials']['use_yolo'] = self.materials_yolo_var.get()
        self.config.config['materials']['detection_threshold'] = self.materials_threshold_var.get()
        self.config.config['materials']['yolo_model_path'] = self.materials_yolo_path_var.get()
        self.config.config['anpr']['enabled'] = self.anpr_enabled_var.get()
        self.config.config['anpr']['yolo_model_path'] = self.anpr_yolo_var.get()
        self.config.config['anpr']['ocr_model_path'] = self.anpr_ocr_var.get()
        self.config.config['anpr']['detection_confidence_threshold'] = self.anpr_confidence_var.get()
        self.config.config['anpr']['use_separate_yolo'] = self.anpr_separate_var.get()
        self.config.config['anpr']['recognize_vehicles'] = self.anpr_vehicles_var.get()
        self.config.save_config()
        if self.on_save_callback:
            self.on_save_callback()
        messagebox.showinfo("Успех", "Настройки сохранены")
        self.window.destroy()

class ModernGUI:
    def __init__(self, root, config, database, model_manager):
        self.root = root
        self.config = config
        self.database = database
        self.model_manager = model_manager
        self.stream_threads = {}
        self.stream_configs = self.config.config['streams']
        self.save_path = self.config.config.get('save_path', 'saved_frames')
        os.makedirs(self.save_path, exist_ok=True)
        save_settings = self.config.config.get('save_settings', {})
        self.save_every_n_frame = save_settings.get('save_every_n_frame', 5)
        self.max_frames_per_event = save_settings.get('max_frames_per_event', 50)
        performance_settings = self.config.config.get('performance', {})
        self.ui_update_interval = performance_settings.get('ui_update_interval', 33)
        self.frame_skip_factor = performance_settings.get('frame_skip_factor', 1)
        self.fullscreen_windows = {}
        self.video_cells = []
        self.events_history = deque(maxlen=100)
        self.global_trigger_person = tk.BooleanVar(value=True)
        self.global_trigger_vehicle = tk.BooleanVar(value=True)
        self.global_trigger_motion = tk.BooleanVar(value=True)
        self.selected_stream_id = None
        self.selected_cell_idx = None
        self.yolo_available = model_manager.is_yolo_available() if model_manager else False
        self.anpr_enabled = config.config.get('anpr', {}).get('enabled', False)
        self.materials_enabled = config.config.get('materials', {}).get('enabled', True)
        self.materials_yolo_enabled = config.config.get('materials', {}).get('use_yolo', True)
        self.context_menu = tk.Menu(root, tearoff=0)
        self.context_menu.add_command(label="➕ Добавить поток", command=self.add_stream_to_cell)
        self.context_menu.add_command(label="✏️ Редактировать поток", command=self.edit_selected_stream)
        self.context_menu.add_command(label="🗑️ Удалить поток", command=self.remove_selected_stream)
        self.context_menu.add_separator()
        self.context_menu.add_command(label="📺 Полноэкранный режим", command=self.open_fullscreen)
        self.setup_styles()
        self.setup_ui()
        self.load_streams_to_cells()
        if hasattr(self.model_manager, 'get_device_info'):
            device_info = self.model_manager.get_device_info()
            if device_info:
                device_name = device_info.get('device_info', {}).get('name', 'CPU')
                yolo_version = device_info.get('yolo_version', 'N/A')
                yolo_status = "✅" if self.yolo_available else "❌"
                anpr_status = "✅" if self.anpr_enabled else "🚫"
                materials_status = "✅" if self.materials_enabled else "🚫"
                materials_yolo_status = "✅" if self.materials_yolo_enabled else "🚫"
                messagebox.showinfo("Информация об устройстве",
                                  f"Используется устройство: {device_name}\n"
                                  f"Тип: {device_info.get('device_key', 'cpu').upper()}\n"
                                  f"YOLO v26 версия: {yolo_version} {yolo_status}\n"
                                  f"ANPR система: {anpr_status}\n"
                                  f"Материалы: {materials_status}\n"
                                  f"YOLO для материалов: {materials_yolo_status}\n\n"
                                  f"Статус YOLO v26: {'Доступен' if self.yolo_available else 'Не доступен'}\n"
                                  f"Статус ANPR: {'Включена' if self.anpr_enabled else 'Выключена'}\n"
                                  f"Статус материалов: {'Включено' if self.materials_enabled else 'Выключено'}\n"
                                  f"Статус YOLO для материалов: {'Включено' if self.materials_yolo_enabled else 'Выключено'}\n"
                                  f"Путь сохранения: {self.save_path}")
        self._last_stats_update = 0
        self.update_ui()

    def setup_styles(self):
        style = ttk.Style()
        style.theme_use('clam')
        self.bg_color = '#f0f2f5'
        self.panel_bg = '#ffffff'
        self.fg_color = '#1a1a1a'
        self.accent_color = '#3b82f6'
        self.success_color = '#10b981'
        self.warning_color = '#f59e0b'
        self.error_color = '#ef4444'
        style.configure('Main.TFrame', background=self.bg_color)
        style.configure('Panel.TFrame', background=self.panel_bg, relief='flat', borderwidth=0)

    def setup_ui(self):
        self.root.configure(bg=self.bg_color)
        self.root.geometry("1400x900")
        anpr_status = "✅" if self.anpr_enabled else "🚫"
        materials_status = "✅" if self.materials_enabled else "🚫"
        materials_yolo_status = "✅" if self.materials_yolo_enabled else "🚫"
        self.root.title(f"AI Видеоанализатор PRO с Multi-GPU поддержкой | YOLOv26: {'✅' if self.yolo_available else '❌'} | ANPR: {anpr_status} | Материалы: {materials_status} | YOLO мат: {materials_yolo_status} | Оптимизированная версия | Сохранение: {self.save_path}")
        main_container = tk.Frame(self.root, bg=self.bg_color)
        main_container.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        self.setup_top_panel(main_container)
        content_frame = tk.Frame(main_container, bg=self.bg_color)
        content_frame.pack(fill=tk.BOTH, expand=True, pady=5)
        left_panel = self.setup_left_panel(content_frame)
        center_panel = self.setup_center_panel(content_frame)
        right_panel = self.setup_right_panel(content_frame)
        self.setup_status_bar(main_container)

    def setup_top_panel(self, parent):
        top_frame = tk.Frame(parent, bg=self.panel_bg, height=60)
        top_frame.pack(fill=tk.X, pady=(0, 5))
        top_frame.pack_propagate(False)
        title_container = tk.Frame(top_frame, bg=self.panel_bg)
        title_container.pack(side=tk.LEFT, fill=tk.Y, padx=20, pady=10)
        icon_label = tk.Label(title_container, text="⚡",
                             font=('Segoe UI', 24),
                             bg=self.panel_bg, fg=self.accent_color)
        icon_label.pack(side=tk.LEFT, padx=(0, 10))
        yolo_status = "✅" if self.yolo_available else "❌"
        anpr_status = "✅" if self.anpr_enabled else "🚫"
        materials_status = "✅" if self.materials_enabled else "🚫"
        materials_yolo_status = "✅" if self.materials_yolo_enabled else "🚫"
        title_text = tk.Label(title_container, text=f"AI Видеоанализатор PRO с Multi-GPU | YOLOv26 {yolo_status} | ANPR {anpr_status} | Материалы {materials_status} | YOLO мат {materials_yolo_status} | Оптимизировано",
                             font=('Segoe UI', 16, 'bold'),
                             bg=self.panel_bg, fg=self.fg_color)
        title_text.pack(side=tk.LEFT)
        control_frame = tk.Frame(top_frame, bg=self.panel_bg)
        control_frame.pack(side=tk.RIGHT, padx=20, pady=10)
        tk.Button(control_frame, text="⚙️ Настройки",
                 command=self.open_settings,
                 bg=self.accent_color, fg='white',
                 font=('Segoe UI', 9, 'bold'),
                 padx=8, pady=4).pack(side=tk.LEFT, padx=2)
        tk.Button(control_frame, text="🏗️ Материалы",
                 command=self.manage_materials,
                 bg=self.warning_color, fg='white',
                 font=('Segoe UI', 9, 'bold'),
                 padx=8, pady=4).pack(side=tk.LEFT, padx=2)
        tk.Button(control_frame, text="📁 Открыть папку сохранения",
                 command=self.open_save_folder,
                 bg=self.warning_color, fg='white',
                 font=('Segoe UI', 9, 'bold'),
                 padx=8, pady=4).pack(side=tk.LEFT, padx=2)
        self.stats_label = tk.Label(top_frame,
                                   text=f"Режим оптимизированный | ANPR: {'Вкл' if self.anpr_enabled else 'Выкл'} | Материалы: {'Вкл' if self.materials_enabled else 'Выкл'} | YOLO мат: {'Вкл' if self.materials_yolo_enabled else 'Выкл'} | Сохранение: {self.save_path}",
                                   font=('Segoe UI', 10),
                                   bg=self.panel_bg, fg=self.success_color)
        self.stats_label.pack(side=tk.RIGHT, padx=20, pady=5)

    def manage_materials(self):
        SettingsWindow(self.root, self.config, self.on_settings_saved, self.model_manager)

    def open_save_folder(self):
        if os.path.exists(self.save_path):
            os.startfile(self.save_path) if os.name == 'nt' else os.system(f'xdg-open "{self.save_path}"')
        else:
            messagebox.showwarning("Папка не найдена", f"Папка сохранения не существует: {self.save_path}")

    def setup_left_panel(self, parent):
        left_panel = tk.Frame(parent, width=280, bg=self.panel_bg)
        left_panel.pack(side=tk.LEFT, fill=tk.BOTH, padx=(0, 5))
        streams_frame = tk.LabelFrame(left_panel, text="📹 Потоки",
                                     bg=self.panel_bg, fg=self.fg_color,
                                     font=('Segoe UI', 10, 'bold'),
                                     padx=8, pady=8)
        streams_frame.pack(fill=tk.X, pady=(0, 8), padx=5)
        tk.Button(streams_frame, text="➕ Добавить поток",
                 command=self.open_add_stream_dialog,
                 bg=self.accent_color, fg='white',
                 font=('Segoe UI', 8, 'bold'),
                 padx=6, pady=3,
                 relief=tk.FLAT).pack(fill=tk.X, pady=2)
        tk.Button(streams_frame, text="▶ Запустить все",
                 command=self.start_all_streams,
                 bg=self.success_color, fg='white',
                 font=('Segoe UI', 8, 'bold'),
                 padx=6, pady=3,
                 relief=tk.FLAT).pack(fill=tk.X, pady=2)
        tk.Button(streams_frame, text="⏹ Остановить все",
                 command=self.stop_all_streams,
                 bg=self.error_color, fg='white',
                 font=('Segoe UI', 8, 'bold'),
                 padx=6, pady=3,
                 relief=tk.FLAT).pack(fill=tk.X, pady=2)
        self.selected_stream_info = tk.Label(streams_frame,
                                           text="Выбран поток: Нет",
                                           font=('Segoe UI', 8),
                                           bg=self.panel_bg, fg=self.accent_color)
        self.selected_stream_info.pack(fill=tk.X, pady=(5, 5))
        manage_frame = tk.Frame(streams_frame, bg=self.panel_bg)
        manage_frame.pack(fill=tk.X, pady=5)
        buttons = [
            ("✏️", self.edit_selected_stream, self.accent_color),
            ("🔄", self.restart_selected_stream, self.warning_color),
            ("🗑️", self.remove_selected_stream, self.error_color),
            ("📺", self.open_fullscreen, self.success_color)
        ]
        for text, command, color in buttons:
            btn = tk.Button(manage_frame, text=text,
                           command=command,
                           bg=color, fg='white',
                           font=('Segoe UI', 9),
                           padx=8, pady=3,
                           relief=tk.FLAT)
            btn.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=1)
        db_frame = tk.LabelFrame(left_panel, text="🗄️ База данных",
                                bg=self.panel_bg, fg=self.fg_color,
                                font=('Segoe UI', 10, 'bold'),
                                padx=8, pady=8)
        db_frame.pack(fill=tk.X, pady=(0, 8), padx=5)
        tk.Button(db_frame, text="👤 Добавить лицо",
                 command=self.add_face,
                 bg=self.accent_color, fg='white',
                 font=('Segoe UI', 8, 'bold'),
                 padx=6, pady=3,
                 relief=tk.FLAT).pack(fill=tk.X, pady=2)
        tk.Button(db_frame, text="🚗 Добавить номер",
                 command=self.add_plate,
                 bg=self.accent_color, fg='white',
                 font=('Segoe UI', 8, 'bold'),
                 padx=6, pady=3,
                 relief=tk.FLAT).pack(fill=tk.X, pady=2)
        tk.Button(db_frame, text="🏗️ Добавить материал",
                 command=self.add_material,
                 bg=self.accent_color, fg='white',
                 font=('Segoe UI', 8, 'bold'),
                 padx=6, pady=3,
                 relief=tk.FLAT).pack(fill=tk.X, pady=2)
        tk.Button(db_frame, text="📊 Статистика БД",
                 command=self.show_db_stats,
                 bg=self.success_color, fg='white',
                 font=('Segoe UI', 8, 'bold'),
                 padx=6, pady=3,
                 relief=tk.FLAT).pack(fill=tk.X, pady=2)
        return left_panel

    def setup_center_panel(self, parent):
        center_panel = tk.Frame(parent, bg=self.panel_bg)
        center_panel.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=5)
        info_frame = tk.Frame(center_panel, bg=self.panel_bg, height=30)
        info_frame.pack(fill=tk.X, pady=(0, 5))
        info_frame.pack_propagate(False)
        yolo_status = "✅" if self.yolo_available else "❌"
        anpr_status = "✅" if self.anpr_enabled else "🚫"
        materials_status = "✅" if self.materials_enabled else "🚫"
        materials_yolo_status = "✅" if self.materials_yolo_enabled else "🚫"
        self.info_label = tk.Label(info_frame,
                                   text=f"Оптимизированный режим | Потоков: 0 | Событий: 0 | YOLOv26: {yolo_status} | ANPR: {anpr_status} | Материалы: {materials_status} | YOLO мат: {materials_yolo_status} | Сохранение: {self.save_path}",
                                   font=('Segoe UI', 10),
                                   bg=self.panel_bg, fg=self.fg_color)
        self.info_label.pack(side=tk.LEFT, padx=10, pady=5)
        self.video_grid = tk.Frame(center_panel, bg=self.panel_bg)
        self.video_grid.pack(fill=tk.BOTH, expand=True)
        self.create_video_cells()
        return center_panel

    def create_video_cells(self):
        self.video_cells = []
        for i in range(20):
            row = i // 5
            col = i % 5
            cell_frame = tk.Frame(self.video_grid,
                                bg=self.panel_bg,
                                highlightbackground='#e5e7eb',
                                highlightthickness=1)
            cell_frame.grid(row=row, column=col, padx=3, pady=3, sticky=tk.NSEW)
            cell_frame.grid_propagate(False)
            cell_frame.config(width=200, height=180)
            canvas = tk.Canvas(cell_frame, width=196, height=130,
                              bg='#f3f4f6', highlightthickness=0)
            canvas.place(x=2, y=2)
            canvas.current_image = None
            status_label = tk.Label(cell_frame, text="Неактивно",
                                   font=('Segoe UI', 8),
                                   bg=self.panel_bg, fg=self.error_color)
            status_label.place(x=5, y=140)
            camera_label = tk.Label(cell_frame, text=f"Камера {i+1}",
                                   font=('Segoe UI', 8, 'bold'),
                                   bg=self.panel_bg, fg=self.fg_color)
            camera_label.place(x=5, y=160)
            canvas.bind('<Button-3>', lambda e, idx=i: self.show_context_menu(e, idx))
            canvas.bind('<Button-1>', lambda e, idx=i: self.select_stream_by_cell(idx))
            canvas.bind('<Double-Button-1>', lambda e, idx=i: self.toggle_fullscreen_view(idx))
            self.video_cells.append({
                'frame': cell_frame,
                'canvas': canvas,
                'status': status_label,
                'camera_label': camera_label,
                'current_image': None,
                'stream_id': None,
                'last_update': 0,
                'is_selected': False
            })
            self.video_grid.rowconfigure(row, weight=1)
            self.video_grid.columnconfigure(col, weight=1)

    def setup_right_panel(self, parent):
        right_panel = tk.Frame(parent, width=350, bg=self.panel_bg)
        right_panel.pack(side=tk.RIGHT, fill=tk.BOTH)
        header_frame = tk.Frame(right_panel, bg=self.panel_bg, height=40)
        header_frame.pack(fill=tk.X, pady=(0, 5))
        header_frame.pack_propagate(False)
        title_label = tk.Label(header_frame, text="📋 Последние события",
                              font=('Segoe UI', 12, 'bold'),
                              bg=self.panel_bg, fg=self.fg_color)
        title_label.pack(side=tk.LEFT, padx=10, pady=5)
        tk.Button(header_frame, text="🧹",
                 command=self.clear_events,
                 bg=self.warning_color, fg='white',
                 font=('Segoe UI', 10),
                 padx=8, pady=4,
                 relief=tk.FLAT).pack(side=tk.RIGHT, padx=10, pady=5)
        events_container = tk.Frame(right_panel, bg=self.panel_bg)
        events_container.pack(fill=tk.BOTH, expand=True)
        self.events_canvas = tk.Canvas(events_container, bg=self.panel_bg,
                                      highlightthickness=0)
        scrollbar = ttk.Scrollbar(events_container, orient=tk.VERTICAL,
                                 command=self.events_canvas.yview)
        self.events_canvas.configure(yscrollcommand=scrollbar.set)
        self.events_frame = tk.Frame(self.events_canvas, bg=self.panel_bg)
        self.events_canvas_window = self.events_canvas.create_window((0, 0),
                                                                    window=self.events_frame,
                                                                    anchor=tk.NW,
                                                                    width=340)
        self.events_canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        self.events_frame.bind('<Configure>', self.on_events_frame_configure)
        self.events_canvas.bind('<Configure>', self.on_events_canvas_configure)
        return right_panel

    def setup_status_bar(self, parent):
        status_frame = tk.Frame(parent, bg='#e5e7eb', height=25)
        status_frame.pack(side=tk.BOTTOM, fill=tk.X, pady=(5, 0))
        status_frame.pack_propagate(False)
        self.status_var = tk.StringVar(value=f"Оптимизированный режим: UI обновление каждые {self.ui_update_interval} мс (~{1000//self.ui_update_interval} FPS) | ANPR: {'Вкл' if self.anpr_enabled else 'Выкл'} | Материалы: {'Вкл' if self.materials_enabled else 'Выкл'} | YOLO мат: {'Вкл' if self.materials_yolo_enabled else 'Выкл'} | Сохранение: {self.save_path}")
        status_bar = tk.Label(status_frame, textvariable=self.status_var,
                             anchor=tk.W, bg='#e5e7eb', fg=self.fg_color,
                             font=('Segoe UI', 9))
        status_bar.pack(side=tk.LEFT, padx=10)
        self.system_info_var = tk.StringVar(value="CPU: -- | Память: --")
        system_label = tk.Label(status_frame, textvariable=self.system_info_var,
                               anchor=tk.E, bg='#e5e7eb', fg=self.fg_color,
                               font=('Segoe UI', 9))
        system_label.pack(side=tk.RIGHT, padx=10)
        anpr_status_text = "✅" if self.anpr_enabled else "🚫"
        materials_status_text = "✅" if self.materials_enabled else "🚫"
        materials_yolo_status_text = "✅" if self.materials_yolo_enabled else "🚫"
        self.device_status_var = tk.StringVar(value=f"YOLOv26: -- | ANPR: {anpr_status_text} | Материалы: {materials_status_text} | YOLO мат: {materials_yolo_status_text} | UI FPS: ~30")
        device_label = tk.Label(status_frame, textvariable=self.device_status_var,
                               anchor=tk.CENTER, bg='#e5e7eb', fg=self.accent_color,
                               font=('Segoe UI', 9))
        device_label.pack(side=tk.RIGHT, padx=20)

    def open_settings(self):
        SettingsWindow(self.root, self.config, self.on_settings_saved, self.model_manager)

    def on_settings_saved(self):
        save_settings = self.config.config.get('save_settings', {})
        self.save_every_n_frame = save_settings.get('save_every_n_frame', 5)
        self.max_frames_per_event = save_settings.get('max_frames_per_event', 50)
        performance_settings = self.config.config.get('performance', {})
        self.ui_update_interval = performance_settings.get('ui_update_interval', 33)
        self.frame_skip_factor = performance_settings.get('frame_skip_factor', 1)
        self.save_path = self.config.config.get('save_path', 'saved_frames')
        self.anpr_enabled = self.config.config.get('anpr', {}).get('enabled', False)
        self.materials_enabled = self.config.config.get('materials', {}).get('enabled', True)
        self.materials_yolo_enabled = self.config.config.get('materials', {}).get('use_yolo', True)
        os.makedirs(self.save_path, exist_ok=True)
        self.status_var.set(f"Оптимизированный режим: UI обновление каждые {self.ui_update_interval} мс (~{1000//self.ui_update_interval} FPS) | ANPR: {'Вкл' if self.anpr_enabled else 'Выкл'} | Материалы: {'Вкл' if self.materials_enabled else 'Выкл'} | YOLO мат: {'Вкл' if self.materials_yolo_enabled else 'Выкл'} | Сохранение: {self.save_path}")
        for stream_id, thread in self.stream_threads.items():
            thread.detector.set_save_path(self.save_path)
            thread.detector.set_save_settings(self.save_every_n_frame, self.max_frames_per_event)
            thread.set_frame_skip_factor(self.frame_skip_factor)
            thread.detector.settings['confidence_threshold'] = self.config.config['detection'].get('confidence_threshold', 0.5)
            if hasattr(thread.detector, 'anpr_enabled'):
                thread.detector.anpr_enabled = self.anpr_enabled
                if self.anpr_enabled:
                    thread.detector.anpr_pipeline = self.model_manager.get_anpr_pipeline()
            if hasattr(thread.detector, 'materials_enabled'):
                thread.detector.materials_enabled = self.materials_enabled
                thread.detector.use_materials_yolo = self.materials_yolo_enabled
                thread.detector.materials_list = self.config.get_materials()
                thread.detector.material_detection_threshold = self.config.config['materials'].get('detection_threshold', 0.5)
                if self.materials_yolo_enabled:
                    thread.detector.materials_yolo_model = self.model_manager.get_materials_yolo_model()
        anpr_status = "✅" if self.anpr_enabled else "🚫"
        materials_status = "✅" if self.materials_enabled else "🚫"
        materials_yolo_status = "✅" if self.materials_yolo_enabled else "🚫"
        messagebox.showinfo("Успех", f"Настройки применены\nПуть сохранения: {self.save_path}\nANPR система: {'Включена' if self.anpr_enabled else 'Выключена'} {anpr_status}\nСистема материалов: {'Включена' if self.materials_enabled else 'Выключена'} {materials_status}\nYOLO для материалов: {'Включено' if self.materials_yolo_enabled else 'Выключено'} {materials_yolo_status}")

    def on_events_frame_configure(self, event):
        self.events_canvas.configure(scrollregion=self.events_canvas.bbox("all"))

    def on_events_canvas_configure(self, event):
        self.events_canvas.itemconfig(self.events_canvas_window, width=event.width)

    def show_context_menu(self, event, cell_idx):
        self.select_stream_by_cell(cell_idx)
        self.context_menu.post(event.x_root, event.y_root)

    def open_add_stream_dialog(self):
        dialog = StreamConfigDialog(self.root)
        self.root.wait_window(dialog)
        if dialog.result:
            self.add_stream_from_dialog(dialog.result)

    def add_stream_from_dialog(self, stream_data):
        if self.selected_cell_idx is None:
            used_cells = [s.get('cell_index', -1) for s in self.stream_configs]
            cell_index = 0
            while cell_index in used_cells and cell_index < 20:
                cell_index += 1
            if cell_index >= 20:
                messagebox.showerror("Ошибка", "Достигнут лимит в 20 потоков")
                return
        else:
            cell_index = self.selected_cell_idx
        existing_ids = [s['id'] for s in self.stream_configs]
        stream_id = generate_stream_id(existing_ids)
        stream_config = {
            'id': stream_id,
            'name': stream_data['name'],
            'url': stream_data['url'],
            'fps': stream_data['fps'],
            'ip': stream_data['ip'],
            'port': stream_data['port'],
            'username': stream_data['username'],
            'password': stream_data['password'],
            'path': stream_data['path'],
            'detection_type': stream_data['detection_type'],
            'detection_level': stream_data['detection_level'],
            'cell_index': cell_index,
            'triggers': stream_data['triggers'],
            'connection_type': stream_data.get('connection_type', 'rtsp')
        }
        self.stream_configs.append(stream_config)
        self.config.config['streams'] = self.stream_configs
        self.config.save_config()
        if cell_index < len(self.video_cells):
            cell = self.video_cells[cell_index]
            cell['camera_label'].config(text=stream_data['name'])
            cell['stream_id'] = stream_id
            cell['status'].config(text="Готов к запуску", fg=self.warning_color)
        anpr_status = "✅" if self.anpr_enabled else "🚫"
        materials_status = "✅" if self.materials_enabled else "🚫"
        materials_yolo_status = "✅" if self.materials_yolo_enabled else "🚫"
        messagebox.showinfo("Успех", f"Поток '{stream_data['name']}' добавлен в ячейку {cell_index + 1}\nПуть сохранения: {self.save_path}\nANPR система: {'Включена' if self.anpr_enabled else 'Выключена'} {anpr_status}\nСистема материалов: {'Включена' if self.materials_enabled else 'Выключена'} {materials_status}\nYOLO для материалов: {'Включено' if self.materials_yolo_enabled else 'Выключено'} {materials_yolo_status}")

    def select_stream_by_cell(self, cell_idx):
        if cell_idx < len(self.video_cells):
            cell = self.video_cells[cell_idx]
            stream_id = cell['stream_id']
            if self.selected_cell_idx is not None and self.selected_cell_idx < len(self.video_cells):
                prev_cell = self.video_cells[self.selected_cell_idx]
                prev_cell['frame'].config(highlightbackground='#e5e7eb')
                prev_cell['is_selected'] = False
            if stream_id:
                cell['frame'].config(highlightbackground=self.accent_color, highlightthickness=2)
                cell['is_selected'] = True
                self.selected_stream_id = stream_id
                self.selected_cell_idx = cell_idx
                stream_config = next((s for s in self.stream_configs if s['id'] == stream_id), None)
                if stream_config:
                    stream_name = stream_config['name']
                    cell_index = stream_config.get('cell_index', cell_idx) + 1
                    self.selected_stream_info.config(
                        text=f"Выбран поток: {stream_name} (Ячейка {cell_index})",
                        fg=self.accent_color
                    )
            else:
                cell['frame'].config(highlightbackground=self.accent_color, highlightthickness=2)
                cell['is_selected'] = True
                self.selected_stream_id = None
                self.selected_cell_idx = cell_idx
                self.selected_stream_info.config(
                    text=f"Пустая ячейка {cell_idx + 1} (готова для добавления потока)",
                    fg=self.warning_color
                )

    def edit_selected_stream(self):
        if not self.selected_stream_id:
            messagebox.showwarning("Предупреждение", "Сначала выберите поток кликом по ячейке")
            return
        stream_id = self.selected_stream_id
        stream_config = next((s for s in self.stream_configs if s['id'] == stream_id), None)
        if not stream_config:
            messagebox.showerror("Ошибка", "Выбранный поток не найден")
            return
        if stream_id in self.stream_threads:
            self.stop_stream(stream_id)
        if stream_id in self.fullscreen_windows:
            self.fullscreen_windows[stream_id].close()
        dialog = StreamConfigDialog(self.root, {
            'name': stream_config['name'],
            'ip': stream_config.get('ip', '192.168.1.100'),
            'port': stream_config.get('port', '554'),
            'username': stream_config.get('username', 'admin'),
            'password': stream_config.get('password', '123456'),
            'path': stream_config.get('path', '/stream'),
            'fps': stream_config.get('fps', 15),
            'detection_type': stream_config.get('detection_type', 'yolo'),
            'detection_level': stream_config.get('detection_level', 'medium'),
            'triggers': stream_config.get('triggers', {
                'person': True,
                'vehicle': True,
                'motion': True
            }),
            'connection_type': stream_config.get('connection_type', 'rtsp')
        })
        self.root.wait_window(dialog)
        if dialog.result:
            for i, stream in enumerate(self.stream_configs):
                if stream['id'] == stream_id:
                    self.stream_configs[i] = {
                        'id': stream_id,
                        'name': dialog.result['name'],
                        'url': dialog.result['url'],
                        'fps': dialog.result['fps'],
                        'ip': dialog.result['ip'],
                        'port': dialog.result['port'],
                        'username': dialog.result['username'],
                        'password': dialog.result['password'],
                        'path': dialog.result['path'],
                        'detection_type': dialog.result['detection_type'],
                        'detection_level': dialog.result['detection_level'],
                        'cell_index': stream.get('cell_index', self.selected_cell_idx),
                        'triggers': dialog.result['triggers'],
                        'connection_type': dialog.result.get('connection_type', 'rtsp')
                    }
                    break
            if self.selected_cell_idx is not None:
                cell = self.video_cells[self.selected_cell_idx]
                cell['camera_label'].config(text=dialog.result['name'])
            cell_index = self.selected_cell_idx + 1
            self.selected_stream_info.config(
                text=f"Выбран поток: {dialog.result['name']} (Ячейка {cell_index})",
                fg=self.accent_color
            )
            self.config.config['streams'] = self.stream_configs
            self.config.save_config()
            anpr_status = "✅" if self.anpr_enabled else "🚫"
            materials_status = "✅" if self.materials_enabled else "🚫"
            materials_yolo_status = "✅" if self.materials_yolo_enabled else "🚫"
            messagebox.showinfo("Успех", f"Поток '{dialog.result['name']}' обновлен\nANPR система: {'Включена' if self.anpr_enabled else 'Выключена'} {anpr_status}\nСистема материалов: {'Включена' if self.materials_enabled else 'Выключена'} {materials_status}\nYOLO для материалов: {'Включено' if self.materials_yolo_enabled else 'Выключено'} {materials_yolo_status}")

    def restart_selected_stream(self):
        if not self.selected_stream_id:
            messagebox.showwarning("Предупреждение", "Сначала выберите поток кликом по ячейке")
            return
        stream_id = self.selected_stream_id
        self.stop_stream(stream_id)
        time.sleep(1)
        stream_config = next((s for s in self.stream_configs if s['id'] == stream_id), None)
        if stream_config:
            self.start_stream(stream_config)
            messagebox.showinfo("Успех", f"Поток перезапущен")

    def remove_selected_stream(self):
        if not self.selected_stream_id:
            messagebox.showwarning("Предупреждение", "Сначала выберите поток кликом по ячейке")
            return
        stream_id = self.selected_stream_id
        stream_config = next((s for s in self.stream_configs if s['id'] == stream_id), None)
        if not stream_config:
            return
        stream_name = stream_config['name']
        if messagebox.askyesno("Подтверждение", f"Удалить поток '{stream_name}'?"):
            self.stop_stream(stream_id)
            if stream_id in self.fullscreen_windows:
                self.fullscreen_windows[stream_id].close()
            if self.selected_cell_idx is not None:
                cell = self.video_cells[self.selected_cell_idx]
                cell['stream_id'] = None
                cell['camera_label'].config(text=f"Камера {self.selected_cell_idx + 1}")
                cell['status'].config(text="Неактивно", fg=self.error_color)
                cell['frame'].config(highlightbackground='#e5e7eb', highlightthickness=1)
            self.stream_configs = [s for s in self.stream_configs if s['id'] != stream_id]
            self.config.config['streams'] = self.stream_configs
            self.config.save_config()
            self.selected_stream_id = None
            self.selected_cell_idx = None
            self.selected_stream_info.config(
                text="Выбран поток: Нет",
                fg='#64748b'
            )
            messagebox.showinfo("Успех", f"Поток '{stream_name}' удален")

    def add_stream_to_cell(self):
        if self.selected_cell_idx is None:
            messagebox.showwarning("Предупреждение", "Сначала выберите ячейку кликом по ней")
            return
        cell = self.video_cells[self.selected_cell_idx]
        if cell['stream_id']:
            response = messagebox.askyesno("Подтверждение",
                                          f"Ячейка уже содержит поток. Заменить его?")
            if not response:
                return
        dialog = StreamConfigDialog(self.root)
        self.root.wait_window(dialog)
        if dialog.result:
            self.add_stream_from_dialog(dialog.result)

    def open_fullscreen(self):
        if not self.selected_stream_id:
            messagebox.showwarning("Предупреждение", "Сначала выберите поток кликом по ячейке")
            return
        stream_id = self.selected_stream_id
        stream_config = next((s for s in self.stream_configs if s['id'] == stream_id), None)
        if not stream_config:
            return
        if stream_id in self.fullscreen_windows:
            self.fullscreen_windows[stream_id].close()
            return
        viewer = FullscreenViewer(self, stream_config['name'], stream_id)
        self.fullscreen_windows[stream_id] = viewer

    def toggle_fullscreen_view(self, cell_idx):
        if cell_idx < len(self.video_cells):
            stream_id = self.video_cells[cell_idx]['stream_id']
            if stream_id:
                if stream_id in self.fullscreen_windows:
                    self.fullscreen_windows[stream_id].close()
                else:
                    stream_config = next((s for s in self.stream_configs if s['id'] == stream_id), None)
                    if stream_config:
                        viewer = FullscreenViewer(self, stream_config['name'], stream_id)
                        self.fullscreen_windows[stream_id] = viewer

    def add_face(self):
        filename = filedialog.askopenfilename(
            title="Выберите фото лица",
            filetypes=[("Изображения", "*.jpg *.jpeg *.png *.bmp")]
        )
        if not filename:
            return
        name = simpledialog.askstring("Добавить лицо", "Введите имя человека:")
        if not name:
            return
        position = simpledialog.askstring("Добавить лицо", "Введите должность (опционально):")
        face_id = self.database.add_face(name, filename, position or "")
        messagebox.showinfo("Успех", f"Лицо '{name}' добавлено в базу данных (ID: {face_id})")

    def add_plate(self):
        filename = filedialog.askopenfilename(
            title="Выберите фото номера",
            filetypes=[("Изображения", "*.jpg *.jpeg *.png *.bmp")]
        )
        if not filename:
            return
        plate_number = simpledialog.askstring("Добавить номер", "Введите номер:")
        if not plate_number:
            return
        owner = simpledialog.askstring("Добавить номер", "Введите владельца (опционально):")
        plate_id = self.database.add_plate(plate_number, filename, owner or "")
        messagebox.showinfo("Успех", f"Номер '{plate_number}' добавлен в базу данных (ID: {plate_id})")

    def add_material(self):
        filename = filedialog.askopenfilename(
            title="Выберите фото материала",
            filetypes=[("Изображения", "*.jpg *.jpeg *.png *.bmp")]
        )
        if not filename:
            return
        material_name = simpledialog.askstring("Добавить материал", "Введите название материала:")
        if not material_name:
            return
        description = simpledialog.askstring("Добавить материал", "Введите описание (опционально):")
        material_id = self.database.add_material(material_name, filename, description or "")
        messagebox.showinfo("Успех", f"Материал '{material_name}' добавлен в базу данных (ID: {material_id})")
        if not self.config.add_material(material_name):
            messagebox.showinfo("Информация", f"Материал '{material_name}' добавлен в список для распознавания")

    def show_db_stats(self):
        stats = {
            "👤 Лица": len(self.database.faces),
            "🚗 Номера": len(self.database.plates),
            "🚙 Автомобили": len(self.database.vehicles),
            "🐾 Животные": len(self.database.animals),
            "🏗️ Материалы": len(self.database.materials)
        }
        message = "📊 СТАТИСТИКА БАЗЫ ДАННЫХ\n\n"
        for key, value in stats.items():
            message += f"{key}: {value}\n"
        messagebox.showinfo("Статистика БД", message)

    def clear_events(self):
        if not self.events_history:
            return
        if messagebox.askyesno("Подтверждение", "Очистить все события?"):
            for widget in self.events_frame.winfo_children():
                widget.destroy()
            self.events_history.clear()

    def load_streams_to_cells(self):
        for stream in self.stream_configs:
            cell_index = stream.get('cell_index', 0)
            if cell_index < len(self.video_cells):
                cell = self.video_cells[cell_index]
                cell['stream_id'] = stream['id']
                cell['camera_label'].config(text=stream['name'])
                cell['status'].config(text="Готов к запуску", fg=self.warning_color)

    def start_all_streams(self):
        for stream_config in self.stream_configs:
            self.start_stream(stream_config)

    def start_stream(self, stream_config):
        stream_id = stream_config['id']
        if stream_id in self.stream_threads:
            thread = self.stream_threads[stream_id]
            if thread.is_running():
                return
            else:
                del self.stream_threads[stream_id]
        thread = VideoStreamThread(
            stream_config,
            self.handle_event,
            self.database,
            self.model_manager,
            self.save_path,
            self.save_every_n_frame,
            self.max_frames_per_event
        )
        thread.set_frame_skip_factor(self.frame_skip_factor)
        self.stream_threads[stream_id] = thread
        thread.start()
        cell_index = stream_config.get('cell_index', 0)
        if cell_index < len(self.video_cells):
            self.video_cells[cell_index]['status'].config(text="Запуск...", fg=self.warning_color)
        if thread.wait_for_initialization(timeout=5.0):
            print(f"✅ Поток {stream_config['name']} успешно инициализирован")
        else:
            print(f"⚠️ Поток {stream_config['name']} не инициализировался вовремя")

    def stop_all_streams(self):
        stream_ids = list(self.stream_threads.keys())
        for stream_id in stream_ids:
            self.stop_stream(stream_id)
        for stream_id in stream_ids:
            if stream_id in self.stream_threads:
                thread = self.stream_threads[stream_id]
                if thread.is_alive():
                    thread.join(timeout=2.0)

    def stop_stream(self, stream_id):
        if stream_id in self.stream_threads:
            thread = self.stream_threads[stream_id]
            try:
                thread.stop()
                if thread.is_alive():
                    thread.join(timeout=2.0)
                del self.stream_threads[stream_id]
                for cell in self.video_cells:
                    if cell['stream_id'] == stream_id:
                        cell['status'].config(text="Остановлен", fg=self.error_color)
                        break
                print(f"✅ Поток {stream_id} остановлен")
            except Exception as e:
                print(f"Ошибка остановки потока {stream_id}: {e}")
                if stream_id in self.stream_threads:
                    del self.stream_threads[stream_id]

    def handle_event(self, event):
        self.events_history.append(event)
        self.add_event_to_panel(event)

    def add_event_to_panel(self, event):
        timestamp = event['timestamp'].strftime("%H:%M:%S")
        if event['event_type'] == 'face_recognized':
            icon = "👤"
            color = "#10b981"
            desc = "Распознано лицо"
        elif event['event_type'] == 'person_detected':
            icon = "👤"
            color = "#f59e0b"
            data = event.get('data', {})
            persons = data.get('persons', 0)
            faces = data.get('faces', 0)
            desc = f"Обнаружены люди: {persons} чел, лица: {faces}"
        elif event['event_type'] == 'vehicle_detected':
            icon = "🚗"
            color = "#8b5cf6"
            data = event.get('data', {})
            vehicles = data.get('vehicles', 0)
            plates = data.get('plates', 0)
            materials = data.get('materials', 0)
            desc = f"Обнаружен транспорт: {vehicles} ед, номеров: {plates}"
            if materials > 0:
                desc += f", материалов: {materials}"
        elif event['event_type'] == 'plate_recognized':
            icon = "🔢"
            color = "#06b6d4"
            data = event.get('data', {})
            plate_text = data.get('plate_text', '')
            confidence = data.get('confidence', 0)
            bbox = data.get('bbox', [0, 0, 0, 0])
            w, h = bbox[2], bbox[3] if len(bbox) >= 4 else (0, 0)
            desc = f"Распознан номер: {plate_text} (размер: {w}x{h}, уверенность: {confidence:.2f})"
        elif event['event_type'] == 'material_recognized':
            icon = "🏗️"
            color = "#ec4899"
            data = event.get('data', {})
            material_name = data.get('material_name', '')
            confidence = data.get('confidence', 0)
            vehicle_type = data.get('vehicle_type', 'грузовик')
            bbox = data.get('bbox', [0, 0, 0, 0])
            w, h = bbox[2], bbox[3] if len(bbox) >= 4 else (0, 0)
            desc = f"Распознан материал: {material_name} (уверенность: {confidence:.0%}) в {vehicle_type} (размер: {w}x{h})"
        elif event['event_type'] == 'motion_detected':
            icon = "🎥"
            color = "#ef4444"
            data = event.get('data', {})
            motion_count = data.get('motion_count', 0)
            desc = f"Обнаружено движение: {motion_count} срабатываний"
        else:
            icon = "ℹ️"
            color = "#64748b"
            desc = "Событие обнаружения"
        triggers = event.get('triggers', {})
        trigger_info = ""
        if triggers:
            active = []
            if triggers.get('person'):
                active.append("👤")
            if triggers.get('vehicle'):
                active.append("🚗")
            if triggers.get('motion'):
                active.append("🎥")
            if active:
                trigger_info = f" [Триггеры: {' '.join(active)}]"
        trigger_stats = event.get('data', {}).get('trigger_stats', {})
        stats_info = ""
        if trigger_stats:
            stats_parts = []
            if trigger_stats.get('person', 0) > 0:
                stats_parts.append(f"👤:{trigger_stats['person']}")
            if trigger_stats.get('vehicle', 0) > 0:
                stats_parts.append(f"🚗:{trigger_stats['vehicle']}")
            if trigger_stats.get('motion', 0) > 0:
                stats_parts.append(f"🎥:{trigger_stats['motion']}")
            if trigger_stats.get('ignored', 0) > 0:
                stats_parts.append(f"🚫:{trigger_stats['ignored']}")
            if stats_parts:
                stats_info = f" [Стат: {' '.join(stats_parts)}]"
        anpr_info = ""
        if event['event_type'] == 'plate_recognized':
            plate_text = event.get('data', {}).get('plate_text', '')
            confidence = event.get('data', {}).get('confidence', 0)
            bbox = event.get('data', {}).get('bbox', [0, 0, 0, 0])
            w, h = bbox[2], bbox[3] if len(bbox) >= 4 else (0, 0)
            anpr_info = f" [Номер: {plate_text} (размер: {w}x{h}, уверенность: {confidence:.2f})]"
        elif event['event_type'] == 'vehicle_detected':
            data = event.get('data', {})
            recognized_plates = data.get('recognized_plates', 0)
            if recognized_plates > 0:
                anpr_info = f" [ANPR: {recognized_plates} распознано]"
        materials_info = ""
        if event['event_type'] == 'material_recognized':
            material_name = event.get('data', {}).get('material_name', '')
            confidence = event.get('data', {}).get('confidence', 0)
            vehicle_type = event.get('data', {}).get('vehicle_type', 'грузовик')
            bbox = event.get('data', {}).get('bbox', [0, 0, 0, 0])
            w, h = bbox[2], bbox[3] if len(bbox) >= 4 else (0, 0)
            materials_info = f" [Материал: {material_name} (уверенность: {confidence:.0%}) в {vehicle_type} (размер: {w}x{h})]"
        elif event['event_type'] == 'vehicle_detected':
            data = event.get('data', {})
            recognized_materials = data.get('recognized_materials', 0)
            if recognized_materials > 0:
                materials_info = f" [Материалы: {recognized_materials} распознано]"
        materials_yolo_info = ""
        if event.get('materials_yolo_available', False):
            materials_yolo_info = " [YOLO мат: ✅]"
        message = f"[{timestamp}] {icon} {event['stream_name']}: {desc}{trigger_info}{stats_info}{anpr_info}{materials_info}{materials_yolo_info}"
        event_frame = tk.Frame(self.events_frame, bg=self.panel_bg)
        event_frame.pack(fill=tk.X, pady=2)
        color_bar = tk.Frame(event_frame, bg=color, width=4)
        color_bar.pack(side=tk.LEFT, fill=tk.Y, padx=(0, 8))
        event_label = tk.Label(event_frame, text=message,
                              anchor='w', justify='left',
                              wraplength=320, bg=self.panel_bg, fg=self.fg_color,
                              font=('Segoe UI', 9))
        event_label.pack(side=tk.LEFT, fill=tk.X, expand=True)
        self.events_canvas.yview_moveto(1.0)
        if len(self.events_frame.winfo_children()) > 50:
            for widget in self.events_frame.winfo_children()[:10]:
                widget.destroy()

    def update_ui(self):
        try:
            current_time = time.time()
            stream_to_cell = {}
            for i, cell in enumerate(self.video_cells):
                if cell['stream_id']:
                    stream_to_cell[cell['stream_id']] = i
            update_stats = current_time - self._last_stats_update >= 1.0
            for stream_id, thread in list(self.stream_threads.items()):
                if stream_id in stream_to_cell:
                    cell_idx = stream_to_cell[stream_id]
                    cell = self.video_cells[cell_idx]
                    if not thread.is_alive():
                        print(f"⚠️ Поток {stream_id} мертв, удаляем")
                        del self.stream_threads[stream_id]
                        cell['status'].config(text="Ошибка", fg=self.error_color)
                        continue
                    if current_time - cell['last_update'] > (self.ui_update_interval / 1000.0):
                        frame = thread.get_frame()
                        if frame is not None:
                            try:
                                frame_resized = cv2.resize(frame, (196, 130))
                                frame_rgb = cv2.cvtColor(frame_resized, cv2.COLOR_BGR2RGB)
                                img = Image.fromarray(frame_rgb)
                                if cell['current_image'] is None:
                                    imgtk = ImageTk.PhotoImage(image=img)
                                    cell['canvas'].create_image(0, 0, anchor=tk.NW, image=imgtk)
                                    cell['current_image'] = imgtk
                                else:
                                    cell['current_image'].paste(img)
                                cell['last_update'] = current_time
                            except Exception as e:
                                if cell['current_image'] is None:
                                    cell['canvas'].delete("all")
                                    cell['canvas'].create_rectangle(0, 0, 196, 130, fill='#f3f4f6')
                                    cell['canvas'].create_text(98, 65, text="Нет кадра",
                                                              font=('Segoe UI', 10), fill='#94a3b8')
                    if update_stats:
                        stats = thread.stats
                        status_text = stats['status']
                        if stats['status'] == 'running':
                            color = self.success_color
                        elif stats['status'] == 'stopped':
                            color = self.error_color
                        elif 'error' in stats['status']:
                            color = self.error_color
                        else:
                            color = self.warning_color
                        triggers = stats.get('triggers', {})
                        trigger_icons = ""
                        if triggers.get('person', True):
                            trigger_icons += "👤"
                        if triggers.get('vehicle', True):
                            trigger_icons += "🚗"
                        if triggers.get('motion', True):
                            trigger_icons += "🎥"
                        elapsed = current_time - stats.get('start_time', current_time)
                        fps = stats.get('frames_processed', 0) / max(1, elapsed)
                        yolo_version = stats.get('yolo_version', 'N/A')
                        yolo_available = stats.get('yolo_available', False)
                        yolo_status = "✅" if yolo_available else "❌"
                        anpr_enabled = stats.get('anpr_enabled', False)
                        anpr_status = "✅" if anpr_enabled else "🚫"
                        materials_enabled = stats.get('materials_enabled', False)
                        materials_status = "✅" if materials_enabled else "🚫"
                        materials_yolo_enabled = stats.get('materials_yolo_enabled', False)
                        materials_yolo_status = "✅" if materials_yolo_enabled else "🚫"
                        trigger_stats_text = ""
                        if hasattr(thread, 'detector') and hasattr(thread.detector, 'trigger_stats'):
                            trigger_stats = thread.detector.trigger_stats
                            trigger_parts = []
                            if trigger_stats.get('person', 0) > 0:
                                trigger_parts.append(f"P:{trigger_stats['person']}")
                            if trigger_stats.get('vehicle', 0) > 0:
                                trigger_parts.append(f"V:{trigger_stats['vehicle']}")
                            if trigger_stats.get('motion', 0) > 0:
                                trigger_parts.append(f"M:{trigger_stats['motion']}")
                            if trigger_parts:
                                trigger_stats_text = f" | {' '.join(trigger_parts)}"
                        anpr_stats_text = ""
                        if hasattr(thread.detector, 'stats') and 'recognized_plates' in thread.detector.stats:
                            recognized_plates = thread.detector.stats.get('recognized_plates', 0)
                            if recognized_plates > 0:
                                anpr_stats_text = f" | ANPR:{recognized_plates}"
                        materials_stats_text = ""
                        if hasattr(thread.detector, 'stats') and 'recognized_materials' in thread.detector.stats:
                            recognized_materials = thread.detector.stats.get('recognized_materials', 0)
                            if recognized_materials > 0:
                                materials_stats_text = f" | Мат:{recognized_materials} ({thread.detector.stats.get('materials_detected', 0)} всего)"
                        materials_yolo_stats_text = ""
                        if hasattr(thread.detector, 'materials_yolo_model') and thread.detector.materials_yolo_model:
                            materials_yolo_stats_text = " | YOLO мат:✅"
                        else:
                            materials_yolo_stats_text = " | YOLO мат:🚫"
                        cell['status'].config(
                            text=f"YOLOv26:{yolo_version}{yolo_status} | ANPR:{anpr_status}{anpr_stats_text} | Мат:{materials_status}{materials_stats_text}{materials_yolo_stats_text} | {status_text[:10]} | {fps:.1f}FPS | {trigger_icons}{trigger_stats_text}",
                            foreground=color
                        )
            for stream_id, viewer in list(self.fullscreen_windows.items()):
                if not viewer.window.winfo_exists():
                    del self.fullscreen_windows[stream_id]
            if update_stats:
                active_streams = len(self.stream_threads)
                total_frames = sum(t.stats.get('frames_processed', 0)
                                  for t in self.stream_threads.values())
                detection_types = {}
                trigger_stats_total = {'person': 0, 'vehicle': 0, 'motion': 0, 'ignored': 0}
                yolo_available_count = 0
                anpr_enabled_count = 0
                materials_enabled_count = 0
                materials_yolo_enabled_count = 0
                total_recognized_plates = 0
                total_plates_detected = 0
                total_recognized_materials = 0
                total_materials_detected = 0
                for thread in self.stream_threads.values():
                    dt = thread.stats.get('detection_type', 'unknown')
                    detection_types[dt] = detection_types.get(dt, 0) + 1
                    yolo_available = thread.stats.get('yolo_available', False)
                    if yolo_available:
                        yolo_available_count += 1
                    anpr_enabled = thread.stats.get('anpr_enabled', False)
                    if anpr_enabled:
                        anpr_enabled_count += 1
                    materials_enabled = thread.stats.get('materials_enabled', False)
                    if materials_enabled:
                        materials_enabled_count += 1
                    materials_yolo_enabled = thread.stats.get('materials_yolo_enabled', False)
                    if materials_yolo_enabled:
                        materials_yolo_enabled_count += 1
                    if hasattr(thread.detector, 'trigger_stats'):
                        for trigger in ['person', 'vehicle', 'motion', 'ignored']:
                            trigger_stats_total[trigger] += thread.detector.trigger_stats.get(trigger, 0)
                    if hasattr(thread.detector, 'stats'):
                        total_recognized_plates += thread.detector.stats.get('recognized_plates', 0)
                        total_plates_detected += thread.detector.stats.get('plates_detected', 0)
                        total_recognized_materials += thread.detector.stats.get('recognized_materials', 0)
                        total_materials_detected += thread.detector.stats.get('materials_detected', 0)
                total_faces = sum(t.detector.stats.get('recognized_faces', 0)
                                 for t in self.stream_threads.values()
                                 if hasattr(t, 'detector'))
                yolo_versions = {}
                for thread in self.stream_threads.values():
                    yolo_ver = thread.stats.get('yolo_version', 'N/A')
                    yolo_available = thread.stats.get('yolo_available', False)
                    yolo_status = "✅" if yolo_available else "❌"
                    key = f"{yolo_ver}{yolo_status}"
                    yolo_versions[key] = yolo_versions.get(key, 0) + 1
                yolo_str = ""
                for ver, count in yolo_versions.items():
                    if ver != 'N/A❌':
                        yolo_str += f"{ver}:{count} "
                yolo_available_status = "✅" if self.yolo_available else "❌"
                anpr_enabled_status = "✅" if self.anpr_enabled else "🚫"
                materials_enabled_status = "✅" if self.materials_enabled else "🚫"
                materials_yolo_enabled_status = "✅" if self.materials_yolo_enabled else "🚫"
                ui_fps = 1000 // self.ui_update_interval
                self.info_label.config(
                    text=f"Оптимизированный режим | UI: ~{ui_fps} FPS | Потоков: {active_streams} | YOLOv26: {yolo_str}| ANPR: {anpr_enabled_count}/{active_streams} | Материалы: {materials_enabled_count}/{active_streams} | YOLO мат: {materials_yolo_enabled_count}/{active_streams} | Кадров: {total_frames} | "
                         f"Событий: {len(self.events_history)} | Сохранение: {self.save_path} | YOLOv26 глобально: {yolo_available_status} | ANPR глобально: {anpr_enabled_status} | Материалы глобально: {materials_enabled_status} | YOLO мат глобально: {materials_yolo_enabled_status} | Номеров: {total_plates_detected}/{total_recognized_plates} | Материалов: {total_materials_detected}/{total_recognized_materials}"
                )
                status_parts = []
                if trigger_stats_total['person'] > 0:
                    status_parts.append(f"👤: {trigger_stats_total['person']}")
                if trigger_stats_total['vehicle'] > 0:
                    status_parts.append(f"🚗: {trigger_stats_total['vehicle']}")
                if trigger_stats_total['motion'] > 0:
                    status_parts.append(f"🎥: {trigger_stats_total['motion']}")
                if trigger_stats_total['ignored'] > 0:
                    status_parts.append(f"🚫: {trigger_stats_total['ignored']}")
                trigger_info = " | ".join(status_parts) if status_parts else "Триггеры не активны"
                status_text = f"Оптимизированный режим | UI: ~{ui_fps} FPS | Лиц: {total_faces} | Номеров: {total_plates_detected}/{total_recognized_plates} | Материалов: {total_materials_detected}/{total_recognized_materials} | ANPR распознано: {total_recognized_plates} | Материалов распознано: {total_recognized_materials} | {trigger_info} | YOLOv26 потоков: {yolo_available_count}/{active_streams} | ANPR потоков: {anpr_enabled_count}/{active_streams} | Материалов потоков: {materials_enabled_count}/{active_streams} | YOLO мат потоков: {materials_yolo_enabled_count}/{active_streams} | Сохранение: {self.save_path}"
                self.status_var.set(status_text)
                if hasattr(self.model_manager, 'get_device_info'):
                    device_info = self.model_manager.get_device_info()
                    if device_info:
                        yolo_version = device_info.get('yolo_version', 'N/A')
                        yolo_initialized = device_info.get('yolo_initialized', False)
                        yolo_status = "✅" if yolo_initialized else "❌"
                        anpr_status = "✅" if self.anpr_enabled else "🚫"
                        materials_status = "✅" if self.materials_enabled else "🚫"
                        materials_yolo_status = "✅" if self.materials_yolo_enabled else "🚫"
                        self.device_status_var.set(f"YOLOv26: {yolo_version} {yolo_status} | ANPR: {anpr_status} | Материалы: {materials_status} | YOLO мат: {materials_yolo_status} | UI FPS: ~{ui_fps} | Сохранение: {self.save_path}")
                self._last_stats_update = current_time
            try:
                cpu_percent = psutil.cpu_percent()
                memory = psutil.virtual_memory()
                memory_percent = memory.percent
                self.system_info_var.set(f"CPU: {cpu_percent:.0f}% | Память: {memory_percent:.0f}%")
            except:
                self.system_info_var.set("CPU: -- | Память: --")
        except Exception as e:
            print(f"Ошибка обновления UI: {e}")
            traceback.print_exc()
        self.root.after(self.ui_update_interval, self.update_ui)

    def on_closing(self):
        for viewer in list(self.fullscreen_windows.values()):
            try:
                viewer.close()
            except:
                pass
        self.stop_all_streams()
        self.config.save_config()
        self.root.destroy()

class FullscreenViewer:
    def __init__(self, parent, stream_name, stream_id):
        self.parent = parent
        self.stream_name = stream_name
        self.stream_id = stream_id
        self.is_closing = False
        self.window = tk.Toplevel(parent.root)
        self.window.title(f"Поток: {stream_name} | Оптимизированный")
        self.window.configure(bg='#0f172a')
        self.window.attributes('-fullscreen', True)
        self.window.protocol("WM_DELETE_WINDOW", self.close)
        self.canvas = tk.Canvas(self.window, bg='#0f172a', highlightthickness=0)
        self.canvas.pack(fill=tk.BOTH, expand=True)
        self.control_frame = tk.Frame(self.window, bg='#000000', bd=0)
        self.control_frame.place(relx=0.5, rely=0.05, anchor='n')
        button_style = {
            'font': ('Segoe UI', 10, 'bold'),
            'padx': 12,
            'pady': 4,
            'bd': 0,
            'highlightthickness': 0,
            'cursor': 'hand2'
        }
        self.close_btn = tk.Button(
            self.control_frame,
            text="✕ Закрыть (ESC)",
            command=self.close,
            bg='#ef4444',
            fg='white',
            activebackground='#dc2626',
            activeforeground='white',
            **button_style
        )
        self.close_btn.pack(side=tk.LEFT, padx=5)
        self.fullscreen_btn = tk.Button(
            self.control_frame,
            text="📺 Обычный режим (F11)",
            command=self.toggle_fullscreen,
            bg='#3b82f6',
            fg='white',
            activebackground='#2563eb',
            activeforeground='white',
            **button_style
        )
        self.fullscreen_btn.pack(side=tk.LEFT, padx=5)
        self.info_frame = tk.Frame(self.window, bg='#000000', bd=0)
        self.info_frame.place(relx=0.5, rely=0.95, anchor='s')
        self.info_label = tk.Label(
            self.info_frame,
            text=f"{stream_name} | Оптимизированный | Загрузка...",
            bg='#000000',
            fg='white',
            font=('Segoe UI', 11),
            padx=12,
            pady=6
        )
        self.info_label.pack()
        self.current_image = None
        self.last_frame_time = 0
        self.fps_counter = deque(maxlen=30)
        self.window.bind('<Escape>', lambda e: self.close())
        self.window.bind('<F11>', self.toggle_fullscreen)
        self.hide_controls_timer = None
        self.controls_visible = True
        self.setup_auto_hide_controls()
        self.window.focus_set()
        self.update_interval = 33
        self.schedule_update()
        self.frame_count = 0
        self.last_fps_update = time.time()

    def setup_auto_hide_controls(self):
        self.window.bind('<Motion>', self.show_controls)
        self.window.bind('<Button>', self.show_controls)
        self.window.bind('<Key>', self.show_controls)
        self.hide_controls_timer = self.window.after(3000, self.hide_controls)

    def show_controls(self, event=None):
        if self.is_closing:
            return
        if not self.controls_visible:
            self.control_frame.place(relx=0.5, rely=0.05, anchor='n')
            self.info_frame.place(relx=0.5, rely=0.95, anchor='s')
            self.controls_visible = True
        if self.hide_controls_timer:
            self.window.after_cancel(self.hide_controls_timer)
        self.hide_controls_timer = self.window.after(3000, self.hide_controls)

    def hide_controls(self):
        if self.is_closing or not self.window.attributes('-fullscreen'):
            return
        if self.controls_visible:
            self.control_frame.place_forget()
            self.info_frame.place_forget()
            self.controls_visible = False

    def schedule_update(self):
        if not self.is_closing:
            self.window.after(self.update_interval, self.update_frame)

    def update_frame(self):
        if self.is_closing:
            return
        try:
            if (hasattr(self.parent, 'stream_threads') and
                self.stream_id in self.parent.stream_threads):
                thread = self.parent.stream_threads[self.stream_id]
                if thread.is_alive():
                    frame = thread.get_frame()
                    if frame is not None:
                        current_time = time.time()
                        self.fps_counter.append(current_time)
                        if current_time - self.last_frame_time > (self.update_interval / 1000.0):
                            self.display_frame(frame)
                            self.last_frame_time = current_time
                else:
                    print(f"⚠️ Поток {self.stream_id} мертв, закрываем окно просмотра")
                    self.close()
                    return
            self.schedule_update()
        except Exception as e:
            print(f"Ошибка обновления кадра: {e}")
            self.schedule_update()

    def display_frame(self, frame):
        try:
            if self.is_closing or not self.window.winfo_exists():
                return
            canvas_width = self.canvas.winfo_width()
            canvas_height = self.canvas.winfo_height()
            if canvas_width <= 1 or canvas_height <= 1:
                canvas_width = self.window.winfo_width()
                canvas_height = self.window.winfo_height()
            if canvas_width <= 1 or canvas_height <= 1:
                canvas_width = 800
                canvas_height = 600
            frame_height, frame_width = frame.shape[:2]
            aspect_ratio = frame_width / frame_height
            if canvas_width / canvas_height > aspect_ratio:
                display_height = canvas_height
                display_width = int(canvas_height * aspect_ratio)
            else:
                display_width = canvas_width
                display_height = int(canvas_width / aspect_ratio)
            frame_resized = cv2.resize(frame, (display_width, display_height))
            frame_rgb = cv2.cvtColor(frame_resized, cv2.COLOR_BGR2RGB)
            img = Image.fromarray(frame_rgb)
            if self.current_image is None:
                self.current_image = ImageTk.PhotoImage(image=img)
                self.canvas.create_image(0, 0, anchor=tk.NW, image=self.current_image)
            else:
                self.current_image.paste(img)
            x_offset = (canvas_width - display_width) // 2
            y_offset = (canvas_height - display_height) // 2
            self.canvas.coords(self.canvas.find_withtag("current")[0], x_offset, y_offset)
            current_time = time.time()
            if current_time - self.last_fps_update > 1.0:
                fps = len(self.fps_counter)
                anpr_info = ""
                materials_info = ""
                materials_yolo_info = ""
                if (hasattr(self.parent, 'stream_threads') and
                    self.stream_id in self.parent.stream_threads):
                    thread = self.parent.stream_threads[self.stream_id]
                    if (hasattr(thread.detector, 'stats') and
                        'recognized_plates' in thread.detector.stats):
                        recognized_plates = thread.detector.stats.get('recognized_plates', 0)
                        if recognized_plates > 0:
                            anpr_info = f" | ANPR: {recognized_plates} номеров распознано"
                    if (hasattr(thread.detector, 'stats') and
                        'recognized_materials' in thread.detector.stats):
                        recognized_materials = thread.detector.stats.get('recognized_materials', 0)
                        if recognized_materials > 0:
                            materials_info = f" | Материалы: {recognized_materials} распознано ({thread.detector.stats.get('materials_detected', 0)} всего)"
                    if hasattr(thread.detector, 'materials_yolo_model') and thread.detector.materials_yolo_model:
                        materials_yolo_info = " | YOLO мат: ✅"
                    else:
                        materials_yolo_info = " | YOLO мат: 🚫"
                self.info_label.config(text=f"{self.stream_name} | Оптимизированный | {fps} FPS{anpr_info}{materials_info}{materials_yolo_info}")
                self.last_fps_update = current_time
        except Exception as e:
            if self.current_image is None:
                self.canvas.delete("all")
                self.canvas.create_text(canvas_width // 2, canvas_height // 2,
                                       text="Ошибка отображения",
                                       fill='white', font=('Segoe UI', 16))

    def toggle_fullscreen(self, event=None):
        is_fullscreen = self.window.attributes('-fullscreen')
        self.window.attributes('-fullscreen', not is_fullscreen)
        if not is_fullscreen:
            self.fullscreen_btn.config(text="📺 Обычный режим (F11)")
        else:
            self.fullscreen_btn.config(text="📺 Полный экран (F11)")

    def close(self):
        self.is_closing = True
        if self.hide_controls_timer:
            self.window.after_cancel(self.hide_controls_timer)
        if self.window.attributes('-fullscreen'):
            self.window.attributes('-fullscreen', False)
        self.window.destroy()

def main():
    global original_stdout, original_stderr
    sys.stdout = original_stdout
    sys.stderr = original_stderr
    root = tk.Tk()
    root.withdraw()
    loading_window = LoadingWindow(root)

    def loading_function():
        try:
            loading_window.update_progress(10, "Загрузка конфигурации")
            config = Config()
            loading_window.update_progress(20, "Инициализация базы данных")
            database = ObjectDatabase(config.config)
            loading_window.update_progress(30, "Определение устройств")
            device_key, device_info = DeviceDetector.get_optimal_device()
            loading_window.update_device_info(device_info)
            loading_window.update_progress(40, "Загрузка моделей")
            model_manager = ModelManager(
                progress_callback=lambda msg, detail="", progress=None:
                    loading_window.update_progress(
                        40 + (progress or 0) * 0.6 if progress else 50,
                        msg,
                        detail
                    ),
                device_preference="auto",
                config=config.config
            )
            model_manager.download_all_models()
            loading_window.update_progress(90, "Запуск оптимизированного интерфейса")
            def start_gui():
                loading_window.close()
                root.deiconify()
                gui = ModernGUI(root, config, database, model_manager)
                root.protocol("WM_DELETE_WINDOW", gui.on_closing)
            root.after(0, start_gui)
        except Exception as e:
            print("Ошибка загрузки:", e)
            traceback.print_exc()
            def safe_close():
                try:
                    loading_window.close()
                    if root.winfo_exists():
                        root.destroy()
                except:
                    pass
            root.after(0, safe_close)

    loading_window.start_loading_in_thread(loading_function)
    root.mainloop()

if __name__ == "__main__":
    main()