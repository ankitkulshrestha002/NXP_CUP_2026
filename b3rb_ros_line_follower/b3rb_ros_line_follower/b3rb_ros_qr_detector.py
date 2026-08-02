# Copyright 2024-2026 NXP
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import CompressedImage
from std_msgs.msg import String
import cv2
import numpy as np

try:
    from pyzbar import pyzbar
except ImportError:
    pyzbar = None

class QRDetector(Node):
    def __init__(self):
        super().__init__('qr_detector')
        self.create_subscription(CompressedImage, '/camera/image_raw/compressed', self.camera_image_callback, 10)
        self.publisher_qr = self.create_publisher(String, '/qr_detection', 10)
        
        self.cv2_detector = cv2.QRCodeDetector()
        
        self.last_qr_reading = None
        self.qr_confirm_count = 0
        self.qr_confirm_threshold = 3
        self.last_published_qr = None
        self.no_qr_counter = 0 
        
        self.get_logger().info("QR Detector Active (Leaky Bucket Debounce + CLAHE Vision).")

    def camera_image_callback(self, message):
        np_arr = np.frombuffer(message.data, np.uint8)
        image = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
        if image is None: return

        # ROI: Look at the top 70% of the image
        height, width = image.shape[:2]
        image_roi = image[0:int(height * 0.70), 0:width]

        qr_data = self.detect_qr_code(image_roi)

        # LEAKY BUCKET DEBOUNCE (Fixes intermittent flashing)
        if qr_data is None:
            self.qr_confirm_count = max(0, self.qr_confirm_count - 1) # Slowly decay, don't instantly reset
            self.no_qr_counter += 1
            if self.no_qr_counter > 40: # ~4 seconds of nothing clears the memory
                self.last_published_qr = None
                self.last_qr_reading = None
            return

        self.no_qr_counter = 0

        if qr_data == self.last_qr_reading:
            self.qr_confirm_count = min(self.qr_confirm_threshold + 2, self.qr_confirm_count + 2)
        else:
            self.last_qr_reading = qr_data
            self.qr_confirm_count = 1

        if self.qr_confirm_count >= self.qr_confirm_threshold and qr_data != self.last_published_qr:
            msg = String()
            msg.data = qr_data
            self.publisher_qr.publish(msg)
            self.last_published_qr = qr_data
            self.get_logger().info(f"✅ CONFIRMED QR DATA: {qr_data}")

    def detect_qr_code(self, image):
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        # ADVANCED PRE-PROCESSING: CLAHE handles Gazebo shadows perfectly
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
        gray = clahe.apply(gray)
        
        if pyzbar is not None:
            decoded = pyzbar.decode(gray)
            if not decoded:
                # 2.0x Upscale for pixelated textures
                upscaled = cv2.resize(gray, None, fx=2.0, fy=2.0, interpolation=cv2.INTER_CUBIC)
                _, thresh = cv2.threshold(upscaled, 100, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
                decoded = pyzbar.decode(thresh)
            
            if decoded:
                best = max(decoded, key=lambda o: o.rect.width * o.rect.height)
                return best.data.decode('utf-8')

        try:
            data, bbox, _ = self.cv2_detector.detectAndDecode(image)
            if bbox is not None and data != "":
                return data
        except Exception:
            pass
        return None

def main(args=None):
    rclpy.init(args=args)
    node = QRDetector()
    try: rclpy.spin(node)
    except KeyboardInterrupt: pass
    finally:
        node.destroy_node()
        if rclpy.ok(): rclpy.shutdown()

if __name__ == '__main__':
    main()