# Copyright 2024-2026 NXP
# Licensed under the Apache License, Version 2.0 

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import CompressedImage
from std_msgs.msg import String
import cv2
import numpy as np
import os

SIGN_LETTERS = ['A', 'B', 'C', 'X', 'Y', 'Z']
TEMPLATE_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'sign_templates')

class ObjectRecognizer(Node):
    """
    ROS 2 Node that detects Green Overhead Signboards.
    Fixes fatal Bug 2: Strips unreliable X-position guessing.
    MUST have templates created in sign_templates/ directory!
    """
    def __init__(self):
        super().__init__('object_recognizer')
        self.subscription_camera = self.create_subscription(
            CompressedImage, '/camera/image_raw/compressed', self.camera_image_callback, 10)
        self.publisher_sign = self.create_publisher(String, '/sign_board_detection', 10)

        # Pre-load templates
        self.templates = {}
        for letter in SIGN_LETTERS:
            path = os.path.join(TEMPLATE_DIR, f'{letter}.png')
            if os.path.exists(path):
                self.templates[letter] = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
            else:
                self.templates[letter] = None
                self.get_logger().warn(f"MISSING TEMPLATE: {path}. Please create it!")
        
        self.get_logger().info("Object Recognizer Active. Looking for Green Signs...")

    def camera_image_callback(self, message):
        np_arr = np.frombuffer(message.data, np.uint8)
        image = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
        if image is None: return

        sign_detected = self.classify_sign(image)
        if sign_detected:
            msg = String()
            msg.data = sign_detected
            self.publisher_sign.publish(msg)

    def match_letter(self, glyph_crop_gray):
        best_letter, best_score = None, -1.0
        for letter, template in self.templates.items():
            if template is None:
                continue
            scale = glyph_crop_gray.shape[0] / template.shape[0]
            resized = cv2.resize(template, None, fx=scale, fy=scale)
            
            if resized.shape[0] > glyph_crop_gray.shape[0] or resized.shape[1] > glyph_crop_gray.shape[1]:
                continue
                
            result = cv2.matchTemplate(glyph_crop_gray, resized, cv2.TM_CCOEFF_NORMED)
            _, score, _, _ = cv2.minMaxLoc(result)
            if score > best_score:
                best_score, best_letter = score, letter
                
        return best_letter if best_score >= 0.5 else None

    def classify_sign(self, image):
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        mask = cv2.inRange(hsv, np.array([35, 50, 50]), np.array([85, 255, 255]))
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        for contour in contours:
            area = cv2.contourArea(contour)
            if area > 1000:
                x, y, w, h = cv2.boundingRect(contour)
                if 1.0 < (float(w) / h) < 10.0:
                    sign_crop = image[y:y+h, x:x+w]
                    gray_crop = cv2.cvtColor(sign_crop, cv2.COLOR_BGR2GRAY)
                    _, white_mask = cv2.threshold(gray_crop, 190, 255, cv2.THRESH_BINARY)

                    # 1. Determine Direction via Centroid of white pixels
                    M = cv2.moments(white_mask)
                    direction = "STRAIGHT"
                    if M["m00"] > 0:
                        cx = int(M["m10"] / M["m00"])
                        if cx < (w / 2.0) - (w * 0.08): direction = "LEFT"
                        elif cx > (w / 2.0) + (w * 0.08): direction = "RIGHT"

                    # 2. Match Letter. DO NOT GUESS if template matching fails!
                    letter = self.match_letter(gray_crop)
                    
                    if letter is None:
                        self.get_logger().warn("Found Sign Direction, but Template Mathing failed to read Letter. Ignoring sign.")
                        return None # Better to let lane-follower default than to inject a lie!

                    return f"{letter}_{direction}"
        return None

def main(args=None):
    rclpy.init(args=args)
    node = ObjectRecognizer()
    try: rclpy.spin(node)
    except KeyboardInterrupt: pass
    finally:
        node.destroy_node()
        if rclpy.ok(): rclpy.shutdown()

if __name__ == '__main__':
    main()