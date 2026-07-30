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
        
        self.last_qr_reading = None
        self.qr_confirm_count = 0
        self.qr_confirm_threshold = 3
        self.last_published_qr = None
        
        self.get_logger().info("QR Detector Active (Debounce Enabled).")

    def camera_image_callback(self, message):
        np_arr = np.frombuffer(message.data, np.uint8)
        image = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
        if image is None: return

        qr_data = self.detect_qr_code(image)

        if qr_data is None:
            self.qr_confirm_count = 0
            self.last_qr_reading = None
            return

        if qr_data == self.last_qr_reading:
            self.qr_confirm_count += 1
        else:
            self.last_qr_reading = qr_data
            self.qr_confirm_count = 1

        if self.qr_confirm_count >= self.qr_confirm_threshold and qr_data != self.last_published_qr:
            msg = String()
            msg.data = qr_data
            self.publisher_qr.publish(msg)
            self.last_published_qr = qr_data
            self.get_logger().info(f"Published QR: {qr_data}")

    def detect_qr_code(self, image):
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        if pyzbar is not None:
            decoded = pyzbar.decode(gray)
            if not decoded:
                enhanced = cv2.equalizeHist(gray)
                upscaled = cv2.resize(enhanced, None, fx=1.5, fy=1.5, interpolation=cv2.INTER_CUBIC)
                decoded = pyzbar.decode(upscaled)
            
            if decoded:
                best = max(decoded, key=lambda o: o.rect.width * o.rect.height)
                return best.data.decode('utf-8')
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