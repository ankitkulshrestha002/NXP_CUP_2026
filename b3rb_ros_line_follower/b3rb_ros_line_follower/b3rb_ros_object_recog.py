# Copyright 2024-2026 NXP
# Licensed under the Apache License, Version 2.0

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import CompressedImage
from std_msgs.msg import String
import cv2
import numpy as np
import os
from collections import deque
from ament_index_python.packages import get_package_share_directory

# WARNING: Ensure 'ultralytics' is allowed by NXP or you have permission!
from ultralytics import YOLO

BOARD_LAYOUT = ['A', 'B', 'C', 'X', 'Y', 'Z']

class ObjectRecognizer(Node):
    """
    ROS 2 Node that detects Green Overhead Signboards.
    Uses YOLO (OBB) model for detecting letters and arrows + Rolling Buffer Logic.
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

        # Use a Rolling History Buffer (Deque) instead of a rigid counter.
        self.history = {letter: deque(maxlen=12) for letter in BOARD_LAYOUT}
        
        # We need 10 confirmations out of the last 12 frames to publish.
        self.CONFIRM_THRESHOLD = 10
        self.letter_classes = set(BOARD_LAYOUT)

        # ====================================================
        # LOAD YOLO MODEL HERE
        # ====================================================
        try:
            package_share_directory = get_package_share_directory('b3rb_ros_line_follower')
            
            # Updated to 'nxpcup.pt' to match your setup.py!
            model_path = os.path.join(package_share_directory, 'best1.pt')
            
            self.get_logger().info(f"Loading YOLO model from: {model_path} on CPU...")
            
            # Load the model. task='obb' is specified for the Oriented Bounding Box model.
            self.model = YOLO(model_path, task='obb')
        except Exception as e:
            self.get_logger().error(f"Failed to load model: {e}")
            self.model = None
            
        self.get_logger().info("🚀 Object Recognizer Active (YOLO + Rolling Buffer Mode).")

    def camera_image_callback(self, message):
        if getattr(self, 'model', None) is None:
            return

        np_arr = np.frombuffer(message.data, np.uint8)
        image = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
        if image is None: return

        # Process frame and get everything detected in THIS specific frame using YOLO
        seen_this_frame = self.process_sign_board(image)

        # Update History Buffer for ALL letters
        for letter in BOARD_LAYOUT:
            dir_seen = seen_this_frame.get(letter, None)
            self.history[letter].append(dir_seen) # Appends None if board wasn't seen
            
            # Count how many times each direction appeared in the last 12 frames
            left_count = self.history[letter].count("LEFT")
            right_count = self.history[letter].count("RIGHT")
            straight_count = self.history[letter].count("STRAIGHT")
            
            best_dir = None
            if left_count >= self.CONFIRM_THRESHOLD:
                best_dir = "LEFT"
            elif right_count >= self.CONFIRM_THRESHOLD:
                best_dir = "RIGHT"
            elif straight_count >= self.CONFIRM_THRESHOLD:
                best_dir = "STRAIGHT"
            
            # If we hit the threshold, publish and clear memory to avoid spamming
            if best_dir is not None:
                msg = String()
                msg.data = f"{letter}_{best_dir}"  # Formats as "A_LEFT" for the LineFollower
                self.publisher_sign.publish(msg)
                self.get_logger().info(f"📡 CONFIRMED SIGN: {letter}_{best_dir} (Rolling Buffer Verified)")
                
                # Clear history for this letter so it doesn't trigger again immediately
                self.history[letter].clear()

    def process_sign_board(self, image):
        """
        Uses YOLO to detect both letters and arrows simultaneously and returns a dictionary.
        """
        detected_in_frame = {}
        detections = []
        
        try:
            # FORCE CPU INFERENCE to bypass the RTX 5050 CUDA crash
            results = self.model.predict(source=image, device='cpu', conf=0.15, verbose=False)
            
            for result in results:
                # Handle OBB (Oriented Bounding Boxes) format
                if hasattr(result, 'obb') and result.obb is not None:
                    for obb in result.obb:
                        x_center = float(obb.xywhr[0][0])
                        class_id = int(obb.cls[0])
                        label = self.model.names[class_id]
                        detections.append({'label': label, 'xc': x_center})
                
                # Handle Standard Bounding Boxes format (Fallback)
                elif hasattr(result, 'boxes') and result.boxes is not None:
                    for box in result.boxes:
                        x_center = float(box.xywh[0][0])
                        class_id = int(box.cls[0])
                        label = self.model.names[class_id]
                        detections.append({'label': label, 'xc': x_center})

        except Exception as e:
            self.get_logger().error(f"Inference error: {e}")
            return detected_in_frame

        # Pair letters and arrows based on closest X-coordinate
        # Using .upper() handles casing issues (e.g., if their YOLO outputs 'Left', it becomes 'LEFT')
        letters = [d for d in detections if d['label'].upper() in self.letter_classes]
        arrows = [d for d in detections if d['label'].upper() in {"LEFT", "RIGHT", "STRAIGHT"}]

        for let in letters:
            best_arrow = None
            min_dist = float('inf')

            for arr in arrows:
                dist = abs(let['xc'] - arr['xc'])
                if dist < min_dist:
                    min_dist = dist
                    best_arrow = arr

            if best_arrow is not None:
                letter_key = let['label'].upper()
                arrow_val = best_arrow['label'].upper()
                detected_in_frame[letter_key] = arrow_val
                
        return detected_in_frame

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