# Copyright 2024-2026 NXP
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import CompressedImage
from std_msgs.msg import String
import cv2
import numpy as np
import os

SIGN_LETTERS = ['A', 'B', 'C', 'X', 'Y', 'Z']
TEMPLATE_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'sign_templates')

# Measured from the actual sign_board_N texture assets
LETTER_TOP_FRAC, LETTER_BOTTOM_FRAC = 0.10, 0.45
ARROW_TOP_FRAC, ARROW_BOTTOM_FRAC = 0.58, 0.85

class ObjectRecognizer(Node):
    def __init__(self):
        super().__init__('object_recognizer')
        self.subscription_camera = self.create_subscription(
            CompressedImage, '/camera/image_raw/compressed', self.camera_image_callback, 10)
        self.publisher_sign = self.create_publisher(String, '/sign_board_detection', 10)
        
        self.templates = {}
        for letter in SIGN_LETTERS:
            path = os.path.join(TEMPLATE_DIR, f'{letter}.png')
            self.templates[letter] = cv2.imread(path, cv2.IMREAD_GRAYSCALE) if os.path.exists(path) else None
            
        self.get_logger().info("Object Recognizer Active (Dynamic Multi-Cell Segmentation).")

    def camera_image_callback(self, message):
        np_arr = np.frombuffer(message.data, np.uint8)
        image = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
        if image is None: return
        
        for sign in self.classify_sign(image):
            msg = String()
            msg.data = sign
            self.publisher_sign.publish(msg)

    def match_letter(self, glyph_crop_gray):
        best_letter, best_score = None, -1.0
        for letter, template in self.templates.items():
            if template is None or glyph_crop_gray.shape[0] < 4 or glyph_crop_gray.shape[1] < 4:
                continue
            scale = glyph_crop_gray.shape[0] / template.shape[0]
            resized = cv2.resize(template, None, fx=scale, fy=scale)
            if resized.shape[0] > glyph_crop_gray.shape[0] or resized.shape[1] > glyph_crop_gray.shape[1]:
                continue
            result = cv2.matchTemplate(glyph_crop_gray, resized, cv2.TM_CCOEFF_NORMED)
            _, score, _, _ = cv2.minMaxLoc(result)
            if score > best_score:
                best_score, best_letter = score, letter
        return best_letter if best_score >= 0.45 else None

    def find_cells(self, white_mask):
        col_sums = np.sum(white_mask > 0, axis=0)
        is_content = col_sums > (white_mask.shape[0] * 0.02)
        cells, start, in_cell = [], 0, False
        for i, has in enumerate(is_content):
            if has and not in_cell:
                start, in_cell = i, True
            elif not has and in_cell:
                if i - start > 6:
                    cells.append((start, i))
                in_cell = False
        if in_cell and (len(is_content) - start > 6):
            cells.append((start, len(is_content)))
        return cells

    def classify_sign(self, image):
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        mask = cv2.inRange(hsv, np.array([35, 50, 50]), np.array([85, 255, 255]))
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours: return []
        
        panel_contour = max(contours, key=cv2.contourArea)
        if cv2.contourArea(panel_contour) < 1000: return []
        
        x, y, w, h = cv2.boundingRect(panel_contour)
        if (float(w) / h) < 1.0 or (float(w) / h) > 12.0: return []

        panel_gray = cv2.cvtColor(image[y:y+h, x:x+w], cv2.COLOR_BGR2GRAY)
        _, white_mask = cv2.threshold(panel_gray, 190, 255, cv2.THRESH_BINARY)

        letter_y0, letter_y1 = int(h * LETTER_TOP_FRAC), int(h * LETTER_BOTTOM_FRAC)
        arrow_y0, arrow_y1 = int(h * ARROW_TOP_FRAC), int(h * ARROW_BOTTOM_FRAC)

        results = []
        for cx0, cx1 in self.find_cells(white_mask):
            letter_crop = panel_gray[letter_y0:letter_y1, cx0:cx1]
            arrow_mask = white_mask[arrow_y0:arrow_y1, cx0:cx1]

            letter = self.match_letter(letter_crop)
            if letter is None: continue

            cell_w = cx1 - cx0
            left_mass = cv2.countNonZero(arrow_mask[:, :cell_w // 2])
            right_mass = cv2.countNonZero(arrow_mask[:, cell_w // 2:])
            direction = "STRAIGHT"
            if left_mass > right_mass * 1.3: direction = "LEFT"
            elif right_mass > left_mass * 1.3: direction = "RIGHT"

            results.append(f"{letter}_{direction}")
            self.get_logger().info(f"Segmented Panel: {letter}_{direction}")
            
        return results

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