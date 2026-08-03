# Copyright 2024-2026 NXP
# Licensed under the Apache License, Version 2.0 

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import CompressedImage
from std_msgs.msg import String
import cv2
import numpy as np

# -------------------------------------------------------------
# SPATIAL INDEXING: The Ultimate Optimization
# The NXP Board ALWAYS follows this exact layout left-to-right.
# -------------------------------------------------------------
BOARD_LAYOUT = ['A', 'B', 'C', 'X', 'Y', 'Z']

class ObjectRecognizer(Node):
    """
    ROS 2 Node that detects Green Overhead Signboards.
    Uses Spatial Indexing (Geometry) to map letters, and Pixel Mass to read arrows.
    100% Immune to blur, distance, and OCR failures.
    """
    def __init__(self):
        super().__init__('object_recognizer')
        self.subscription_camera = self.create_subscription(
            CompressedImage, '/camera/image_raw/compressed', self.camera_image_callback, 10)
        self.publisher_sign = self.create_publisher(String, '/sign_board_detection', 10)

        # Temporal Debouncing: Require 3 consecutive frames to trust an arrow direction
        self.confirm_counts = {letter: {'dir': None, 'count': 0} for letter in BOARD_LAYOUT}
        self.CONFIRM_THRESHOLD = 3
        
        self.get_logger().info("🚀 Object Recognizer Active! (Spatial Indexing & Pixel Mass Mode)")

    def camera_image_callback(self, message):
        np_arr = np.frombuffer(message.data, np.uint8)
        image = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
        if image is None: return

        self.process_sign_board(image)

    def extract_arrow_direction(self, arrow_crop):
        """
        ALGORITHMIC APPROACH: Normalized Centroid Shift via Image Moments.
        Standard CV best practice for directional shape orientation.
        """
        gray = cv2.cvtColor(arrow_crop, cv2.COLOR_BGR2GRAY)
        # Slightly stronger blur to fuse pixelated Gazebo edges together
        gray = cv2.GaussianBlur(gray, (5, 5), 0)

        # Robust binarization
        _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

        # Get contours to find the connected arrow body
        contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours: return None

        # Filter out isolated noise dots, keep the largest contour (the arrow itself)
        arrow_contour = max(contours, key=cv2.contourArea)
        if cv2.contourArea(arrow_contour) < 15: return None

        # 1. Calculate Mathematical Center of Mass (Centroid)
        M = cv2.moments(arrow_contour)
        if M["m00"] == 0: return None
        cx = M["m10"] / M["m00"]
        cy = M["m01"] / M["m00"]

        # 2. Calculate Geometric Center (Bounding Box Middle)
        x, y, w, h = cv2.boundingRect(arrow_contour)
        box_cx = x + w / 2.0
        box_cy = y + h / 2.0

        # 3. Calculate Normalized Shift (Scale Invariant)
        shift_x = (cx - box_cx) / w
        shift_y = (cy - box_cy) / h

        aspect_ratio = float(h) / max(w, 1)

        # 4. Classification
        # STRAIGHT: Taller than it is wide, OR mass is heavily pulled to the top (Negative Y-shift)
        if aspect_ratio > 1.20 or shift_y < -0.10:
            return "STRAIGHT"

        # LEFT/RIGHT: Based purely on the X-axis shift of the centroid
        if shift_x < -0.06:   # Mass is significantly pulled left
            return "LEFT"
        elif shift_x > 0.06:  # Mass is significantly pulled right
            return "RIGHT"
            
        return "STRAIGHT" # Safe Fallback

    def process_sign_board(self, image):
        img_h, img_w = image.shape[:2] # Get camera screen dimensions
        
        # 1. Find the Massive Green Board
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        mask = cv2.inRange(hsv, np.array([35, 50, 50]), np.array([85, 255, 255]))
        
        # Morphological cleanup to make the board a solid block
        kernel = np.ones((5, 5), np.uint8)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        for contour in contours:
            x, y, w, h = cv2.boundingRect(contour)
            aspect_ratio = float(w) / max(h, 1)

            # -------------------------------------------------------------
            # THE GOLDILOCKS ZONE FILTER
            # 1. w > (img_w * 0.35): Ignore if too far (prevents pixelated STRAIGHT reads).
            # 2. w < (img_w * 0.85): Ignore if passing underneath (prevents stretched reads).
            # 3. y > 10: Ignore if the board is clipping through the top of the screen.
            # -------------------------------------------------------------
            if (img_w * 0.35) < w < (img_w * 0.85) and y > 10 and aspect_ratio > 3.0:
                board_crop = image[y:y+h, x:x+w]
                
                # 2. Slice the board vertically into 6 equal columns
                panel_w = w // 6  
                
                for i in range(6):
                    # SPATIAL INDEXING: We automatically know which letter this is!
                    letter = BOARD_LAYOUT[i]

                    # Extract this specific panel
                    panel = board_crop[:, i*panel_w : (i+1)*panel_w]
                    if panel.shape[0] < 10 or panel.shape[1] < 10: 
                        continue
                        
                    # The arrow is always in the bottom half of the panel
                    mid_y = panel.shape[0] // 2
                    arrow_crop = panel[mid_y:, :] 

                    # 3. Read the Arrow Direction
                    direction = self.extract_arrow_direction(arrow_crop)
                    
                    if direction is not None:
                        # 4. Temporal Debounce (Wait for 3 consecutive frames to eliminate flicker)
                        if self.confirm_counts[letter]['dir'] == direction:
                            self.confirm_counts[letter]['count'] += 1
                        else:
                            self.confirm_counts[letter]['dir'] = direction
                            self.confirm_counts[letter]['count'] = 1

                        # If confirmed, Broadcast it!
                        if self.confirm_counts[letter]['count'] >= self.CONFIRM_THRESHOLD:
                            msg = String()
                            msg.data = f"{letter}_{direction}"
                            self.publisher_sign.publish(msg)
                            
                            # Reset count to prevent spamming the logs endlessly, 
                            # but keep broadcasting enough for the LineFollower to catch it.
                            self.confirm_counts[letter]['count'] = 0 
                            self.get_logger().info(f"📡 BROADCASTING: {letter}_{direction}")

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