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

# HSV ranges for cone red/orange stripes (analyzed from actual Gazebo texture)
# Red hue wraps at 0/180, cone reds are H=0-12, and can shift to ~15 under lighting
CONE_RED_LOW_1  = np.array([0, 30, 30])
CONE_RED_HIGH_1 = np.array([15, 255, 255])
CONE_RED_LOW_2  = np.array([165, 30, 30])
CONE_RED_HIGH_2 = np.array([180, 255, 255])

class ObjectRecognizer(Node):
    """
    ROS 2 Node that detects Green Overhead Signboards (via YOLO OBB) and
    red/white construction cones for parking (via HSV color segmentation).
    Camera + LIDAR hybrid parking support.
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

        # Cone detection publisher
        self.publisher_cone = self.create_publisher(
            String,
            '/cone_detection_cam',
            10)

        # Subscribe to mission state to activate cone detection only during parking
        self.create_subscription(
            String,
            '/mission_state',
            self.mission_state_callback,
            10)
        self.is_parking_mode = False

        # Use a Rolling History Buffer (Deque) instead of a rigid counter.
        self.history = {letter: deque(maxlen=12) for letter in BOARD_LAYOUT}
        
        # We need 10 confirmations out of the last 12 frames to publish.
        self.CONFIRM_THRESHOLD = 10
        self.letter_classes = set(BOARD_LAYOUT)

        # Cone detection debounce (2 frames for fast response)
        self.cone_confirm_direction = None
        self.cone_confirm_count = 0
        self.CONE_CONFIRM_THRESHOLD = 2

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
            
        self.get_logger().info("🚀 Object Recognizer Active (YOLO + Cone Detection Mode).")

    def mission_state_callback(self, message):
        """Activate/deactivate cone detection based on mission state."""
        state = message.data.strip().upper()
        was_parking = self.is_parking_mode
        self.is_parking_mode = (state == "EXIT_TO_PARK")
        if self.is_parking_mode and not was_parking:
            self.cone_confirm_direction = None
            self.cone_confirm_count = 0
            self.get_logger().info("🅿️ PARKING MODE — Cone detection enabled.")

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

        # Run cone detection (only active during parking mode)
        self.detect_cones(image)

    def detect_cones(self, image):
        """Detect red/orange construction cones and report LEFT/RIGHT/CENTER."""
        if not self.is_parking_mode:
            return

        img_h, img_w = image.shape[:2]
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

        # Combine both red/orange hue ranges
        mask1 = cv2.inRange(hsv, CONE_RED_LOW_1, CONE_RED_HIGH_1)
        mask2 = cv2.inRange(hsv, CONE_RED_LOW_2, CONE_RED_HIGH_2)
        masak = cv2.bitwise_or(mask1, mask2)

        # Only look at bottom 75% of frame (cones are ground-level)
        top_cutoff = int(img_h * 0.25)
        mask[:top_cutoff, :] = 0

        # Clean up noise
        kernel = np.ones((5, 5), np.uint8)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        # Filter: cones are vertical objects with some minimum area
        cone_contours = []
        for c in contours:
            x, y, w, h = cv2.boundingRect(c)
            area = cv2.contourArea(c)
            # Relaxed: area > 40 (small cones at distance), h > w * 0.8 (roughly vertical)
            if area > 40 and h > w * 0.8:
                cone_contours.append((x + w / 2.0, y, w, h, area))

        if len(cone_contours) < 2:
            # Not enough cones — reset debounce
            self.cone_confirm_count = 0
            self.cone_confirm_direction = None
            return

        # Find average X position of all detected cones
        avg_x = sum(c[0] for c in cone_contours) / len(cone_contours)
        center_x = img_w / 2.0

        # Determine direction
        if avg_x < center_x - img_w * 0.08:
            position = "LEFT"
        elif avg_x > center_x + img_w * 0.08:
            position = "RIGHT"
        else:
            position = "CENTER"

        # Debounce: 2 consecutive frames with same direction
        if self.cone_confirm_direction == position:
            self.cone_confirm_count += 1
        else:
            self.cone_confirm_direction = position
            self.cone_confirm_count = 1

        if self.cone_confirm_count >= self.CONE_CONFIRM_THRESHOLD:
            msg = String()
            msg.data = f"CONES_{position}_{len(cone_contours)}"
            self.publisher_cone.publish(msg)
            self.get_logger().info(
                f"🔶 CONES: {position} ({len(cone_contours)} cones, avg_x={avg_x:.0f}/{img_w})"
            )

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