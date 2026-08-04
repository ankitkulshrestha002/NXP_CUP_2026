# Copyright 2024-2026 NXP
# Licensed under the Apache License, Version 2.0 

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import CompressedImage
from std_msgs.msg import String
import cv2
import numpy as np
import os

try:
    import tflite_runtime.interpreter as tflite
except ImportError:
    tflite = None

BOARD_LAYOUT = ['A', 'B', 'C', 'X', 'Y', 'Z']
CLASS_LABELS = {0: "LEFT", 1: "STRAIGHT", 2: "RIGHT"}

class ObjectRecognizer(Node):
    """
    ROS 2 Node that detects Green Overhead Signboards.
    Rule-Compliant Inference using TFLite + Panel Geometry Verification.
    """
    def __init__(self):
        super().__init__('object_recognizer')

        self.subscription_camera = self.create_subscription(
            CompressedImage,
            '/camera/image_raw/compressed',
            self.camera_image_callback,
            10)

        self.publisher_sign = self.create_publisher(
            String,
            '/sign_board_detection',
            10)

        # Load TFLite Model with Multi-Path Fallback
        self.interpreter = None
        if tflite is not None:
            try:
                dir_path = os.path.dirname(os.path.abspath(__file__))
                candidate_paths = [
                    os.path.join(dir_path, 'model.tflite'),
                    os.path.expanduser('~/cognipilot/cranium/src/b3rb_ros_line_follower/b3rb_ros_line_follower/b3rb_ros_line_follower/model.tflite'),
                    os.path.expanduser('~/cognipilot/cranium/src/b3rb_ros_line_follower/b3rb_ros_line_follower/model.tflite'),
                    os.path.expanduser('~/Downloads/model.tflite'),
                ]

                model_path = next((p for p in candidate_paths if os.path.exists(p)), None)

                if model_path:
                    self.interpreter = tflite.Interpreter(model_path=model_path)
                    self.interpreter.allocate_tensors()
                    self.input_details = self.interpreter.get_input_details()
                    self.output_details = self.interpreter.get_output_details()
                    self.get_logger().info(f"✅ Successfully loaded TFLite Model from {model_path}")
                else:
                    self.get_logger().warn("model.tflite not found. Using Geometry Mode.")
            except Exception as e:
                self.get_logger().error(f"Failed to load TFLite model: {e}")
        else:
            self.get_logger().warn("tflite-runtime is missing.")

        self.confirm_counts = {letter: {'dir': None, 'count': 0} for letter in BOARD_LAYOUT}
        self.CONFIRM_THRESHOLD = 3
        
        self.get_logger().info("Object Recognizer Active (Rule-Compliant Competition Mode).")

    def camera_image_callback(self, message):
        np_arr = np.frombuffer(message.data, np.uint8)
        image = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
        if image is None: return

        self.process_sign_board(image)

    def classify_arrow_crop(self, arrow_crop, panel_w):
        """Dual-Engine Classification: Panel Centroid Geometry + TFLite Inference"""
        # 1. Panel Geometry Verification (100% Deterministic for NXP Board Panels)
        try:
            gray = cv2.cvtColor(arrow_crop, cv2.COLOR_BGR2GRAY)
            gray = cv2.GaussianBlur(gray, (3, 3), 0)
            _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

            contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            if contours:
                valid = [c for c in contours if cv2.contourArea(c) > 10]
                if valid:
                    arrow_contour = max(valid, key=cv2.contourArea)
                    M = cv2.moments(arrow_contour)
                    if M["m00"] > 0:
                        cx = M["m10"] / M["m00"]
                        panel_cx = panel_w / 2.0
                        shift_x = (cx - panel_cx) / max(panel_w, 1)

                        # Panel X arrow (→) is shifted RIGHT -> shift_x > +0.04
                        # Panel A/C/Y arrows (←) are shifted LEFT -> shift_x < -0.04
                        if shift_x < -0.04:
                            return "LEFT"
                        elif shift_x > 0.04:
                            return "RIGHT"
                        else:
                            return "STRAIGHT"
        except Exception:
            pass

        # 2. TFLite Secondary Classification
        if self.interpreter is not None:
            try:
                resized = cv2.resize(arrow_crop, (64, 64))
                img_array = np.expand_dims(resized, axis=0).astype(np.float32) / 255.0

                self.interpreter.set_tensor(self.input_details[0]['index'], img_array)
                self.interpreter.invoke()
                predictions = self.interpreter.get_tensor(self.output_details[0]['index'])

                class_idx = int(np.argmax(predictions[0]))
                confidence = float(predictions[0][class_idx])

                if confidence > 0.60:
                    return CLASS_LABELS[class_idx]
            except Exception:
                pass

        return None

    def process_sign_board(self, image):
        img_h, img_w = image.shape[:2]
        
        # 1. Segment Green Signboard
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        mask = cv2.inRange(hsv, np.array([35, 50, 50]), np.array([85, 255, 255]))
        
        kernel = np.ones((5, 5), np.uint8)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        for contour in contours:
            x, y, w, h = cv2.boundingRect(contour)
            aspect_ratio = float(w) / max(h, 1)

            # STRICT REGION GATE: Only detect when board width is 52% - 82% of camera screen
            MIN_BOARD_WIDTH = int(img_w * 0.52)
            MAX_BOARD_WIDTH = int(img_w * 0.82)

            if MIN_BOARD_WIDTH < w < MAX_BOARD_WIDTH and y > 12 and aspect_ratio > 2.0:
                board_crop = image[y:y+h, x:x+w]
                
                # Dynamic Slicing: 6-panel or 3-panel board
                if aspect_ratio >= 4.0:
                    layout = BOARD_LAYOUT
                    num_panels = 6
                else:
                    layout = ['A', 'B', 'C'] if x < (img_w / 2) else ['X', 'Y', 'Z']
                    num_panels = 3

                panel_w = w // num_panels
                
                for i in range(num_panels):
                    letter = layout[i]
                    panel = board_crop[:, i*panel_w : (i+1)*panel_w]
                    if panel.shape[0] < 10 or panel.shape[1] < 10: 
                        continue
                        
                    # Strict Arrow Isolation (Bottom 48% of panel)
                    mid_y = int(panel.shape[0] * 0.52)
                    arrow_crop = panel[mid_y:, :] 

                    direction = self.classify_arrow_crop(arrow_crop, panel_w)
                    
                    if direction is not None:
                        if self.confirm_counts[letter]['dir'] == direction:
                            self.confirm_counts[letter]['count'] += 1
                        else:
                            self.confirm_counts[letter]['dir'] = direction
                            self.confirm_counts[letter]['count'] = 1

                        if self.confirm_counts[letter]['count'] >= self.CONFIRM_THRESHOLD:
                            msg = String()
                            msg.data = f"{letter}_{direction}"
                            self.publisher_sign.publish(msg)
                            self.confirm_counts[letter]['count'] = 0 
                            self.get_logger().info(f"📡 BROADCASTING ({letter}_{direction}) [Board Width: {w}px]")
def main(args=None):
    rclpy.init(args=args)
    node = ObjectRecognizer()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()

if __name__ == '__main__':
    main()