import os
import cv2
import numpy as np
import mediapipe as mp
import tensorflow as tf
from tensorflow.keras.models import Sequential, Model, load_model
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, LSTM, Dropout, GlobalAveragePooling2D
from ultralytics import YOLO
from sklearn.model_selection import train_test_split # Might not be directly used in Flask app but keep for class definitions
import json
import logging
from datetime import datetime
import pickle # Not directly used in the provided snippet, but keep if part of other functions

# Configure logging for backend (optional, but good for debugging server)
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('backend_surveillance.log'), # Log to a separate file for backend
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class Config:
    """Configuration management for the surveillance system"""
    def __init__(self, config_path="config.json"):
        self.config_path = config_path
        self.default_config = {
            "dataset_path": "dataset", # Not used by Flask, but part of original code
            "models_dir": "models", # Crucial: points to the 'models' subfolder in backend
            "yolo_model": "yolov8m.pt",
            "weapon_classes": ["knife", "gun", "pistol", "rifle", "sword", "firearm"],
            "confidence_threshold": 0.5,
            "sequence_length": 30,
            "input_shape": [128, 128, 3],
            "lstm_epochs": 10,
            "lstm_batch_size": 8,
            "alert_threshold": 0.7,
            "video_source": 0, # No GUI, will be stream based
            "sample_limit": 5000,
            "real_weapon_threshold": 0.8,
            "screen_detection_enabled": True,
            "depth_analysis_enabled": True
        }
        self.config = self.load_config()

    def load_config(self):
        # Attempt to load config from a path relative to the script execution
        current_script_dir = os.path.dirname(os.path.abspath(__file__))
        config_full_path = os.path.join(current_script_dir, self.config_path)

        if os.path.exists(config_full_path):
            try:
                with open(config_full_path, 'r') as f:
                    config = json.load(f)
                merged_config = self.default_config.copy()
                merged_config.update(config)
                return merged_config
            except Exception as e:
                logger.warning(f"Error loading config from {config_full_path}: {e}. Using defaults.")
                return self.default_config
        else:
            self.save_config() # Save default if not found
            return self.default_config

    def save_config(self):
        current_script_dir = os.path.dirname(os.path.abspath(__file__))
        config_full_path = os.path.join(current_script_dir, self.config_path)
        with open(config_full_path, 'w') as f:
            json.dump(self.config, f, indent=4)

    def get(self, key, default=None):
        return self.config.get(key, default)

class RealWeaponClassifier:
    """Classifier to distinguish between real weapons and images/screens"""

    def __init__(self, config):
        self.config = config
        # Ensure models_dir is correctly set relative to the script
        current_script_dir = os.path.dirname(os.path.abspath(__file__))
        self.models_dir = os.path.join(current_script_dir, config.get('models_dir'))
        os.makedirs(self.models_dir, exist_ok=True) # Ensure models directory exists
        self.classifier_model = None
        self.load_or_create_classifier()

    def create_classifier_model(self):
        """Create a CNN model to classify real vs fake weapons"""
        try:
            input_shape = tuple(self.config.get('input_shape'))

            model = Sequential([
                Conv2D(32, (3, 3), activation='relu', input_shape=input_shape),
                MaxPooling2D(2, 2),
                Conv2D(64, (3, 3), activation='relu'),
                MaxPooling2D(2, 2),
                Conv2D(128, (3, 3), activation='relu'),
                MaxPooling2D(2, 2),
                Conv2D(256, (3, 3), activation='relu'),
                GlobalAveragePooling2D(),
                Dense(512, activation='relu'),
                Dropout(0.5),
                Dense(256, activation='relu'),
                Dropout(0.3),
                Dense(128, activation='relu'),
                Dropout(0.2),
                Dense(1, activation='sigmoid')
            ])

            model.compile(
                optimizer='adam',
                loss='binary_crossentropy',
                metrics=['accuracy']
            )

            logger.info("Real weapon classifier model created")
            return model

        except Exception as e:
            logger.error(f"Error creating classifier model: {e}")
            return None

    def load_or_create_classifier(self):
        """Load existing classifier or create new one"""
        classifier_path = os.path.join(self.models_dir, 'real_weapon_classifier.h5')

        if os.path.exists(classifier_path):
            try:
                self.classifier_model = load_model(classifier_path)
                logger.info("Real weapon classifier loaded successfully")
            except Exception as e:
                logger.error(f"Error loading classifier: {e}. Attempting to create new.")
                self.classifier_model = self.create_classifier_model()
        else:
            self.classifier_model = self.create_classifier_model()

    def analyze_weapon_reality(self, frame, bbox):
        """Analyze if detected weapon is real or on screen/photo"""
        if not self.classifier_model:
            return 0.5, "Classifier not available"

        try:
            x1, y1, x2, y2 = bbox
            # Ensure coordinates are within frame bounds
            x1, y1 = max(0, x1), max(0, y1)
            x2, y2 = min(frame.shape[1], x2), min(frame.shape[0], y2)

            weapon_roi = frame[y1:y2, x1:x2]

            if weapon_roi.size == 0 or weapon_roi.shape[0] == 0 or weapon_roi.shape[1] == 0:
                return 0.0, "Invalid ROI (empty or zero dimension)"

            # Resize to model input shape
            input_shape = self.config.get('input_shape')
            weapon_roi_resized = cv2.resize(weapon_roi, (input_shape[1], input_shape[0]))
            weapon_roi_normalized = weapon_roi_resized.astype(np.float32) / 255.0

            # Multiple analysis techniques
            reality_scores = []
            analysis_details = []

            # 1. CNN-based classification
            cnn_input = weapon_roi_normalized[np.newaxis, ...]
            cnn_score = float(self.classifier_model.predict(cnn_input, verbose=0)[0][0])
            reality_scores.append(cnn_score)
            analysis_details.append(f"CNN: {cnn_score:.3f}")

            # 2. Edge analysis (real objects have more natural edges)
            edge_score = self.analyze_edge_patterns(weapon_roi)
            reality_scores.append(edge_score)
            analysis_details.append(f"Edge: {edge_score:.3f}")

            # 3. Texture analysis
            texture_score = self.analyze_texture_reality(weapon_roi)
            reality_scores.append(texture_score)
            analysis_details.append(f"Texture: {texture_score:.3f}")

            # 4. Screen detection
            screen_score = self.detect_screen_patterns(weapon_roi)
            reality_scores.append(1.0 - screen_score)  # Invert screen score
            analysis_details.append(f"Screen: {screen_score:.3f}")

            # Combined score with weights
            weights = [0.4, 0.2, 0.2, 0.2]  # CNN gets highest weight
            final_score = sum(w * s for w, s in zip(weights, reality_scores))

            details = " | ".join(analysis_details)
            return final_score, details

        except Exception as e:
            logger.error(f"Error in weapon reality analysis: {e}")
            return 0.5, f"Analysis error: {str(e)}"

    def analyze_edge_patterns(self, roi):
        """Analyze edge patterns - real objects have more natural edges"""
        try:
            gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
            edges = cv2.Canny(gray, 50, 150)
            edge_density = np.sum(edges > 0) / edges.size
            if 0.05 <= edge_density <= 0.3:
                return min(1.0, edge_density * 3.33)
            else:
                return max(0.0, 1.0 - abs(edge_density - 0.175) * 4)
        except Exception:
            return 0.5

    def analyze_texture_reality(self, roi):
        """Analyze texture patterns - screens have pixel grids, photos have compression artifacts"""
        try:
            gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
            lbp_var = cv2.Laplacian(gray, cv2.CV_64F).var()
            if 100 <= lbp_var <= 2000:
                return min(1.0, lbp_var / 2000)
            else:
                return max(0.0, 1.0 - abs(lbp_var - 1050) / 2000)
        except Exception:
            return 0.5

    def detect_screen_patterns(self, roi):
        """Detect screen patterns like pixel grids or moiré effects"""
        try:
            gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
            f_transform = np.fft.fft2(gray)
            f_shift = np.fft.fftshift(f_transform)
            magnitude_spectrum = np.log(np.abs(f_shift) + 1)
            h, w = magnitude_spectrum.shape
            center_h, center_w = h // 2, w // 2
            grid_score = 0
            for i in range(10, min(50, h // 4)):
                if magnitude_spectrum[center_h + i, center_w] > magnitude_spectrum.mean() + 2 * magnitude_spectrum.std():
                    grid_score += 0.1
                if magnitude_spectrum[center_h, center_w + i] > magnitude_spectrum.mean() + 2 * magnitude_spectrum.std():
                    grid_score += 0.1
            return min(1.0, grid_score)
        except Exception:
            return 0.0

    def save_classifier(self):
        """Save the trained classifier"""
        if self.classifier_model:
            try:
                classifier_path = os.path.join(self.models_dir, 'real_weapon_classifier.h5')
                self.classifier_model.save(classifier_path)
                logger.info("Real weapon classifier saved")
            except Exception as e:
                logger.error(f"Error saving classifier: {e}")

class EnhancedWeaponDetector:
    """Enhanced YOLO-based weapon detection with reality check"""

    def __init__(self, config):
        self.config = config
        self.weapon_classes = config.get('weapon_classes')
        self.confidence_threshold = config.get('confidence_threshold')
        self.real_weapon_threshold = config.get('real_weapon_threshold')
        self.reality_classifier = RealWeaponClassifier(config)

        # Ensure models_dir is correctly set relative to the script
        current_script_dir = os.path.dirname(os.path.abspath(__file__))
        yolo_model_path = os.path.join(current_script_dir, config.get('models_dir'), config.get('yolo_model'))

        try:
            # Use a specific version for loading YOLO model or ensure it's compatible
            # self.yolo_model = YOLO(yolo_model_path, task='detect') # Specify task if needed
            self.yolo_model = YOLO(yolo_model_path)
            logger.info(f"YOLO model loaded successfully from {yolo_model_path}")
        except Exception as e:
            logger.error(f"Failed to load YOLO model from {yolo_model_path}: {e}")
            self.yolo_model = None

    def detect_weapons(self, frame):
        """Enhanced weapon detection with reality verification"""
        detections = []

        if self.yolo_model is None:
            return frame, detections

        try:
            # YOLO expects a BGR image, which OpenCV provides by default
            results = self.yolo_model(frame, verbose=False) # verbose=False to reduce console output
            result = results[0] # Assuming single frame processing

            if result.boxes is not None:
                for box in result.boxes:
                    x1, y1, x2, y2 = map(int, box.xyxy[0])
                    confidence = float(box.conf[0])
                    class_id = int(box.cls[0])
                    label = self.yolo_model.names[class_id].lower()

                    if label in self.weapon_classes and confidence > self.confidence_threshold:
                        reality_score, analysis_details = self.reality_classifier.analyze_weapon_reality(
                            frame, (x1, y1, x2, y2)
                        )

                        is_real_threat = reality_score > self.real_weapon_threshold

                        if is_real_threat:
                            color = (0, 0, 255)  # Red for real weapons (BGR)
                            threat_level = "REAL THREAT"
                        else:
                            color = (0, 165, 255)  # Orange for screen/photo weapons (BGR)
                            threat_level = "SCREEN/PHOTO"

                        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                        label_text = f"{label.upper()}: {confidence:.2f}"
                        threat_text = f"{threat_level} ({reality_score:.2f})"

                        cv2.putText(frame, label_text, (x1, y1 - 30),
                                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
                        cv2.putText(frame, threat_text, (x1, y1 - 10),
                                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)

                        detection_info = {
                            'label': label,
                            'confidence': confidence,
                            'bbox': (x1, y1, x2, y2),
                            'reality_score': reality_score,
                            'is_real_threat': is_real_threat,
                            'analysis_details': analysis_details,
                            'threat_level': threat_level
                        }
                        detections.append(detection_info)

        except Exception as e:
            logger.error(f"Error in weapon detection: {e}")

        return frame, detections

# --- Removed VideoInputManager as it's GUI-based ---
# class VideoInputManager: ... (This class is removed)

class EnhancedAlertSystem:
    """Enhanced alert system with threat classification"""

    def __init__(self, config):
        self.config = config
        self.alert_threshold = config.get('alert_threshold')
        self.alert_log = [] # This will be emitted to frontend
        self.alert_dirs = {
            'real_threats': 'alerts/real_threats',
            'false_positives': 'alerts/false_positives',
            'suspicious_behavior': 'alerts/suspicious_behavior'
        }
        # Create alerts directory structure (relative to script)
        current_script_dir = os.path.dirname(os.path.abspath(__file__))
        for alert_dir_name in self.alert_dirs.values():
            os.makedirs(os.path.join(current_script_dir, alert_dir_name), exist_ok=True)

    # Modified to return alert info instead of just logging/saving
    def trigger_alert(self, frame, alert_type, confidence=None, details=None, threat_level="UNKNOWN"):
        """Enhanced alert system with threat classification"""
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        alert_info = {
            'timestamp': timestamp,
            'type': alert_type,
            'confidence': f"{confidence:.2f}" if confidence is not None else "N/A",
            'details': details if details else "N/A",
            'threat_level': threat_level
        }

        self.alert_log.append(alert_info) # Keep a server-side log

        # Server-side logging
        if threat_level == "REAL THREAT":
            logger.critical(f"🚨 CRITICAL ALERT: {alert_type} - {timestamp}")
        elif threat_level == "SCREEN/PHOTO":
            logger.warning(f"⚠️  POSSIBLE FALSE POSITIVE: {alert_type} - {timestamp}")
        else:
            logger.warning(f"🔔 SECURITY ALERT: {alert_type} - {timestamp}")
        if confidence:
            logger.info(f"Confidence: {confidence:.2f}")
        if details:
            logger.info(f"Details: {details}")

        # Visual alert on frame with color coding (for processed frame output)
        if threat_level == "REAL THREAT":
            alert_color = (0, 0, 255)  # Red (BGR)
            alert_text = f"🚨 CRITICAL: {alert_type.upper()}!"
        elif threat_level == "SCREEN/PHOTO":
            alert_color = (0, 165, 255)  # Orange (BGR)
            alert_text = f"⚠️  POSSIBLE FAKE: {alert_type.upper()}"
        else:
            alert_color = (0, 255, 255)  # Yellow (BGR)
            alert_text = f"🔔 ALERT: {alert_type.upper()}!"

        cv2.putText(frame, alert_text, (10, 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, alert_color, 2)
        if confidence is not None:
            cv2.putText(frame, f"Confidence: {confidence:.2f}", (10, 90),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, alert_color, 2)

        # Save alert frame in appropriate directory
        if threat_level == "REAL THREAT":
            alert_dir = os.path.join(current_script_dir, self.alert_dirs['real_threats'])
        elif threat_level == "SCREEN/PHOTO":
            alert_dir = os.path.join(current_script_dir, self.alert_dirs['false_positives'])
        else:
            alert_dir = os.path.join(current_script_dir, self.alert_dirs['suspicious_behavior'])

        alert_filename = f"alert_{timestamp.replace(':', '-').replace(' ', '_')}.jpg"
        cv2.imwrite(os.path.join(alert_dir, alert_filename), frame)

        return frame, alert_info # Return the alert_info for sending to frontend

class DatasetLoader:
    """Handles dataset loading and preprocessing"""
    # This class will primarily be used for model training, not directly by the server for live detection
    # Kept for completeness of the original script's components
    def __init__(self, config):
        self.config = config

    def load_frame_dataset(self, dataset_path, sample_limit=None):
        if not os.path.exists(dataset_path):
            logger.error(f"Dataset path does not exist: {dataset_path}")
            return [], []
        file_paths = []
        labels = []
        class_dirs = {'train': 1, 'valid': 0}
        for class_name, label in class_dirs.items():
            class_path = os.path.join(dataset_path, class_name)
            if not os.path.exists(class_path):
                logger.warning(f"Class directory not found: {class_path}")
                continue
            class_files = [f for f in os.listdir(class_path)
                          if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
            max_files = sample_limit // len(class_dirs) if sample_limit else len(class_files)
            for file in class_files[:max_files]:
                file_path = os.path.join(class_path, file)
                file_paths.append(file_path)
                labels.append(label)
        logger.info(f"Found {len(file_paths)} image files")
        return file_paths, labels

    def load_image_batch(self, file_paths, batch_size=32):
        h, w = self.config.get('input_shape')[:2]
        for i in range(0, len(file_paths), batch_size):
            batch_paths = file_paths[i:i + batch_size]
            batch_images = []
            for file_path in batch_paths:
                try:
                    image = cv2.imread(file_path)
                    if image is not None:
                        image = cv2.resize(image, (w, h)).astype(np.float32) / 255.0
                        batch_images.append(image)
                except Exception as e:
                    logger.warning(f"Error loading {file_path}: {e}")
                    continue
            if batch_images:
                yield np.array(batch_images)
            else:
                yield np.array([])

class PoseAnalyzer:
    """MediaPipe-based pose analysis"""

    def __init__(self):
        self.mp_pose = mp.solutions.pose
        self.pose = self.mp_pose.Pose(
            static_image_mode=False,
            model_complexity=1,
            enable_segmentation=False,
            min_detection_confidence=0.5
        )
        self.mp_drawing = mp.solutions.drawing_utils

    def analyze_pose(self, frame):
        """Analyze pose and return pose landmarks"""
        try:
            results = self.pose.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            return results.pose_landmarks
        except Exception as e:
            logger.error(f"Error in pose analysis: {e}")
            return None

    def is_suspicious_pose(self, landmarks):
        """Analyze if pose indicates suspicious behavior"""
        if not landmarks:
            return False
        try:
            left_shoulder = landmarks.landmark[self.mp_pose.PoseLandmark.LEFT_SHOULDER]
            right_shoulder = landmarks.landmark[self.mp_pose.PoseLandmark.RIGHT_SHOULDER]
            left_wrist = landmarks.landmark[self.mp_pose.PoseLandmark.LEFT_WRIST]
            right_wrist = landmarks.landmark[self.mp_pose.PoseLandmark.RIGHT_WRIST]

            left_raised = left_wrist.y < left_shoulder.y
            right_raised = right_wrist.y < right_shoulder.y

            return left_raised and right_raised
        except:
            return False

class BehaviorAnalyzer:
    """LSTM-based behavior analysis"""

    def __init__(self, config):
        self.config = config
        self.sequence_length = config.get('sequence_length')
        self.feature_extractor = None
        self.lstm_model = None
        # Ensure models_dir is correctly set relative to the script
        current_script_dir = os.path.dirname(os.path.abspath(__file__))
        self.models_dir = os.path.join(current_script_dir, config.get('models_dir'))
        os.makedirs(self.models_dir, exist_ok=True)

    def create_feature_extractor(self):
        """Create ResNet50-based feature extractor"""
        try:
            input_shape = tuple(self.config.get('input_shape'))
            base_model = tf.keras.applications.ResNet50(
                weights="imagenet",
                include_top=False,
                input_shape=input_shape
            )
            model = Model(inputs=base_model.input, outputs=Flatten()(base_model.output))
            logger.info("Feature extractor created successfully")
            return model
        except Exception as e:
            logger.error(f"Error creating feature extractor: {e}")
            return None

    def create_lstm_model(self, input_shape):
        """Create LSTM model for behavior analysis"""
        try:
            model = Sequential([
                LSTM(64, return_sequences=True, input_shape=input_shape),
                Dropout(0.3),
                LSTM(32, return_sequences=False),
                Dropout(0.3),
                Dense(16, activation='relu'),
                Dense(1, activation='sigmoid')
            ])

            model.compile(
                optimizer='adam',
                loss='binary_crossentropy',
                metrics=['accuracy']
            )

            logger.info("LSTM model created successfully")
            return model
        except Exception as e:
            logger.error(f"Error creating LSTM model: {e}")
            return None

    def save_models(self):
        """Save trained models"""
        try:
            if self.feature_extractor:
                self.feature_extractor.save(os.path.join(self.models_dir, 'feature_extractor.h5'))
            if self.lstm_model:
                self.lstm_model.save(os.path.join(self.models_dir, 'lstm_model.h5'))
            logger.info("Models saved successfully")
        except Exception as e:
                logger.error(f"Error saving models: {e}")

    def load_models(self):
        """Load pre-trained models"""
        try:
            feature_path = os.path.join(self.models_dir, 'feature_extractor.h5')
            lstm_path = os.path.join(self.models_dir, 'lstm_model.h5')

            if os.path.exists(feature_path):
                self.feature_extractor = load_model(feature_path)
                logger.info("Feature extractor loaded successfully")
            else:
                self.feature_extractor = self.create_feature_extractor()

            if os.path.exists(lstm_path):
                self.lstm_model = load_model(lstm_path)
                logger.info("LSTM model loaded successfully")
            else:
                logger.warning("LSTM model not found. Training required.")

        except Exception as e:
            logger.error(f"Error loading models: {e}")
            self.feature_extractor = self.create_feature_extractor()


    def preprocess_dataset(self, dataset_path):
        """Preprocess dataset for LSTM training with memory optimization"""
        if not self.feature_extractor:
            logger.error("Feature extractor not available")
            return None, None

        dataset_loader = DatasetLoader(self.config)
        file_paths, labels = dataset_loader.load_frame_dataset(
            dataset_path,
            self.config.get('sample_limit', 1000)
        )

        if len(file_paths) == 0:
            return None, None

        logger.info(f"Processing {len(file_paths)} images in batches...")

        all_features = []
        batch_size = 16

        for batch_images in dataset_loader.load_image_batch(file_paths, batch_size):
            if len(batch_images) > 0:
                features = self.feature_extractor.predict(batch_images, verbose=0)
                all_features.extend(features)

        if len(all_features) == 0:
            return None, None

        features_array = np.array(all_features)
        labels_array = np.array(labels[:len(features_array)])

        sequences, sequence_labels = self.create_sequences(features_array, labels_array)

        logger.info(f"Created {len(sequences)} sequences for training")
        return sequences, sequence_labels

    def create_sequences(self, features, labels):
        """Create sequences for LSTM training"""
        sequences = []
        sequence_labels = []

        for i in range(len(features) - self.sequence_length + 1):
            seq = features[i:i + self.sequence_length]
            label = labels[i + self.sequence_length - 1]
            sequences.append(seq)
            sequence_labels.append(label)

        return np.array(sequences), np.array(sequence_labels)

    def train_lstm(self, dataset_path):
        """Train LSTM model on dataset"""
        if not self.feature_extractor:
            logger.error("Feature extractor not available")
            return False

        sequences, labels = self.preprocess_dataset(dataset_path)
        if sequences is None or len(sequences) == 0:
            logger.error("Failed to preprocess dataset or no sequences generated.")
            return False

        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            sequences, labels, test_size=0.2, random_state=42
        )

        # Create LSTM model
        input_shape = (self.sequence_length, sequences.shape[2])
        self.lstm_model = self.create_lstm_model(input_shape)

        if not self.lstm_model:
            return False

        # Train model
        try:
            history = self.lstm_model.fit(
                X_train, y_train,
                epochs=self.config.get('lstm_epochs', 10),
                batch_size=self.config.get('lstm_batch_size', 8),
                validation_data=(X_test, y_test),
                verbose=1
            )

            # Save models
            self.save_models()
            logger.info("LSTM training completed successfully")
            return True

        except Exception as e:
            logger.error(f"Error training LSTM: {e}")
            return False

class SecuritySurveillanceSystem:
    """Main surveillance system class with enhanced features"""

    def __init__(self, config_path="config.json"):
        self.config = Config(config_path) # Config uses 'models' relative to backend script
        self.weapon_detector = EnhancedWeaponDetector(self.config)
        self.pose_analyzer = PoseAnalyzer()
        self.behavior_analyzer = BehaviorAnalyzer(self.config)
        self.alert_system = EnhancedAlertSystem(self.config)

        self.frame_sequence = []
        self.is_running = False # Not directly used for server loop but for internal state
        self.current_frame = None # Not directly used like in original GUI version

        # Load models on initialization
        self.behavior_analyzer.load_models()

        logger.info("Security Surveillance System initialized")

    def process_frame_for_web(self, frame):
        """
        Process single frame for threats for web.
        Returns processed frame (as bytes) and a list of alerts.
        """
        processed_frame = frame.copy()
        threats_detected = False
        alerts_triggered = [] # To collect alerts for this frame

        # 1. Enhanced weapon detection with reality check
        weapon_frame, weapon_detections = self.weapon_detector.detect_weapons(processed_frame)
        processed_frame = weapon_frame # Update frame with detections

        if weapon_detections:
            for detection in weapon_detections:
                threat_level = detection['threat_level']
                confidence = detection['confidence']
                reality_score = detection['reality_score']

                # Trigger alert and get alert info back
                frame_with_alert, alert_info = self.alert_system.trigger_alert(
                    processed_frame,
                    f"Weapon detected: {detection['label']}",
                    confidence,
                    f"Reality: {reality_score:.2f} | {detection['analysis_details']}",
                    threat_level
                )
                processed_frame = frame_with_alert # Update frame with alert text
                alerts_triggered.append(alert_info)
                if detection['is_real_threat']:
                    threats_detected = True

        # 2. Pose analysis for suspicious behavior
        pose_landmarks = self.pose_analyzer.analyze_pose(processed_frame)
        if pose_landmarks:
            # Draw pose landmarks (optional for web output, can make frames busy)
            self.pose_analyzer.mp_drawing.draw_landmarks(
                processed_frame, pose_landmarks, self.pose_analyzer.mp_pose.POSE_CONNECTIONS
            )

            if self.pose_analyzer.is_suspicious_pose(pose_landmarks):
                frame_with_alert, alert_info = self.alert_system.trigger_alert(
                    processed_frame,
                    "Suspicious pose detected",
                    confidence=None, # Pose analysis doesn't have a direct confidence score like YOLO
                    threat_level="SUSPICIOUS_BEHAVIOR"
                )
                processed_frame = frame_with_alert
                alerts_triggered.append(alert_info)


        # 3. Behavior analysis (if LSTM model is available)
        if self.behavior_analyzer.lstm_model and self.behavior_analyzer.feature_extractor:
            behavior_alert_info = self.analyze_behavior_sequence_for_web(frame) # Pass original frame for feature extraction
            if behavior_alert_info:
                # Behavior analysis also updates processed_frame internally via alert_system.trigger_alert
                alerts_triggered.append(behavior_alert_info)


        # Encode the processed frame to JPEG for sending over network
        # Ensure the frame is suitable for JPEG encoding (BGR, uint8)
        if processed_frame.dtype != np.uint8:
            processed_frame = (processed_frame * 255).astype(np.uint8)

        ret, buffer = cv2.imencode('.jpg', processed_frame, [int(cv2.IMWRITE_JPEG_QUALITY), 90])
        frame_bytes = buffer.tobytes() if ret else None

        # Return processed frame bytes and any alerts that were triggered
        return frame_bytes, alerts_triggered, threats_detected


    def analyze_behavior_sequence_for_web(self, frame):
        """Analyze behavior using LSTM model and return alert info if triggered"""
        alert_info = None
        try:
            input_shape = self.config.get('input_shape')
            resized_frame = cv2.resize(frame, (input_shape[1], input_shape[0]))
            normalized_frame = resized_frame.astype(np.float32) / 255.0

            features = self.behavior_analyzer.feature_extractor.predict(
                normalized_frame[np.newaxis, ...], verbose=0
            )[0]

            self.frame_sequence.append(features)

            if len(self.frame_sequence) > self.config.get('sequence_length'):
                self.frame_sequence.pop(0)

            if len(self.frame_sequence) == self.config.get('sequence_length'):
                sequence = np.array(self.frame_sequence)[np.newaxis, ...]
                prediction = self.behavior_analyzer.lstm_model.predict(sequence, verbose=0)[0][0]

                if prediction > self.config.get('alert_threshold', 0.7):
                    _, alert_info = self.alert_system.trigger_alert(
                        frame, # Use original frame for alert visuals if desired, or processed one
                        "Suspicious behavior pattern detected",
                        prediction,
                        threat_level="SUSPICIOUS_BEHAVIOR"
                    )
        except Exception as e:
            logger.error(f"Error in behavior analysis: {e}")
        return alert_info

    # --- Removed run_surveillance, show_system_info, cleanup as they are GUI/CLI specific ---
    # def run_surveillance(self): ...
    # def show_system_info(self): ...
    # def cleanup(self): ...
    # def create_sample_config(): ... (if it was part of the class, move it out or remove)

    def train_behavior_model(self, dataset_path):
        """Train the behavior analysis model - can be exposed via API endpoint if needed"""
        logger.info("Starting behavior model training...")
        if self.behavior_analyzer.train_lstm(dataset_path):
            logger.info("Behavior model training completed successfully")
            return True
        else:
            logger.error("Behavior model training failed")
            return False
        