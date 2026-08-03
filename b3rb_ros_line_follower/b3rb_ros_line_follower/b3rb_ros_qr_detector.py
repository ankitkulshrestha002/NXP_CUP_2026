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
        self.subscription_camera = self.create_subscription(
            CompressedImage, '/camera/image_raw/compressed', self.camera_image_callback, 10)
        self.publisher_qr = self.create_publisher(String, '/qr_detection', 10)
        
        # We completely deleted cv2.QRCodeDetector() because it causes the ECI warning!

        self.last_published_qr = None
        self.consecutive_reads = 0
        
        if pyzbar is None:
            self.get_logger().error("❌ CRITICAL: Pyzbar is missing! Run: sudo apt-get install libzbar0 python3-pyzbar")
        else:
            self.get_logger().info("✅ QR Detector Active (Pure Pyzbar Mode).")

    def camera_image_callback(self, message):
        if pyzbar is None: return

        np_arr = np.frombuffer(message.data, np.uint8)
        image = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
        if image is None: return

        # Removed ROI cropping! If the camera is tilted, cropping might cut the QR code in half.
        # We pass the full image to Pyzbar.
        
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        # 1. Try reading the raw grayscale image first
        decoded = pyzbar.decode(gray)
        
        # 2. If it fails due to Gazebo lighting, try a strict Binary Threshold
        if not decoded:
            _, thresh = cv2.threshold(gray, 100, 255, cv2.THRESH_BINARY)
            decoded = pyzbar.decode(thresh)

        if decoded:
            # Find the largest QR code on the screen
            best_qr = max(decoded, key=lambda o: o.rect.width * o.rect.height)
            qr_data = best_qr.data.decode('utf-8')
            
            # Simple Debounce: Prevent spamming the publisher
            # Continuously publish QR detection while in view
            msg = String()
            msg.data = qr_data
            self.publisher_qr.publish(msg)
            self.get_logger().info(f"🚀 PUBLISHED QR DATA: {qr_data}")
        else:
            # Reset if we lose sight of it
            self.consecutive_reads = 0

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