# Copyright 2024-2026 NXP
# Licensed under the Apache License, Version 2.0

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import CompressedImage
from std_msgs.msg import String
import cv2
import numpy as np

BOARD_LAYOUT = ['A', 'B', 'C', 'X', 'Y', 'Z']


class ObjectRecognizer(Node):
    """
    ROS 2 Node that detects Green Overhead Signboards.
    Uses Pure Arrow Mask Geometry + 5-Frame Confirmation Delay.
    """

    def __init__(self):
        super().__init__('object_recognizer')

        self.subscription_camera = self.create_subscription(
            CompressedImage,
            '/camera/image_raw/compressed',
            self.camera_image_callback,
            10
        )

        self.publisher_sign = self.create_publisher(
            String,
            '/sign_board_detection',
            10
        )

        # Require 5 consecutive identical frames before broadcasting (Confirmation Delay)
        self.confirm_counts = {
            letter: {'dir': None, 'count': 0}
            for letter in BOARD_LAYOUT
        }

        self.CONFIRM_THRESHOLD = 5

        self.get_logger().info(
            "🚀 Object Recognizer Active (5-Frame Stability Analysis Mode)."
        )

    def camera_image_callback(self, message):
        np_arr = np.frombuffer(
            message.data,
            np.uint8
        )

        image = cv2.imdecode(
            np_arr,
            cv2.IMREAD_COLOR
        )

        if image is None:
            return

        self.process_sign_board(image)

    def classify_arrow_crop(self, arrow_crop):
        """Pure Arrow Mask Contour Geometry (100% Deterministic for Gazebo Arrows)"""
        try:
            gray = cv2.cvtColor(
                arrow_crop,
                cv2.COLOR_BGR2GRAY
            )

            gray = cv2.GaussianBlur(
                gray,
                (3, 3),
                0
            )

            _, binary = cv2.threshold(
                gray,
                0,
                255,
                cv2.THRESH_BINARY + cv2.THRESH_OTSU
            )

            contours, _ = cv2.findContours(
                binary,
                cv2.RETR_EXTERNAL,
                cv2.CHAIN_APPROX_SIMPLE
            )

            if contours:
                valid = [
                    c for c in contours
                    if cv2.contourArea(c) > 12
                ]

                if valid:
                    arrow_contour = max(
                        valid,
                        key=cv2.contourArea
                    )

                    x, y, w, h = cv2.boundingRect(
                        arrow_contour
                    )

                    if w >= 5 and h >= 5:
                        arrow_mask = binary[
                            y:y+h,
                            x:x+w
                        ]

                        # Vertical check: Straight arrows (↑) are taller than wide
                        if h > w * 1.25:
                            return "STRAIGHT"

                        # Horizontal check: Compare Left Half vs Right Half White Pixel Mass
                        half_w = w // 2

                        left_mass = cv2.countNonZero(
                            arrow_mask[:, :half_w]
                        )

                        right_mass = cv2.countNonZero(
                            arrow_mask[:, half_w:]
                        )

                        if left_mass > right_mass * 1.08:
                            return "LEFT"

                        elif right_mass > left_mass * 1.08:
                            return "RIGHT"

                        else:
                            return "STRAIGHT"

        except Exception:
            pass

        return None

    def process_sign_board(self, image):
        img_h, img_w = image.shape[:2]

        # 1. Segment Green Signboard
        hsv = cv2.cvtColor(
            image,
            cv2.COLOR_BGR2HSV
        )

        mask = cv2.inRange(
            hsv,
            np.array([35, 50, 50]),
            np.array([85, 255, 255])
        )

        kernel = np.ones(
            (5, 5),
            np.uint8
        )

        mask = cv2.morphologyEx(
            mask,
            cv2.MORPH_OPEN,
            kernel
        )

        mask = cv2.morphologyEx(
            mask,
            cv2.MORPH_CLOSE,
            kernel
        )

        contours, _ = cv2.findContours(
            mask,
            cv2.RETR_EXTERNAL,
            cv2.CHAIN_APPROX_SIMPLE
        )

        for contour in contours:
            x, y, w, h = cv2.boundingRect(
                contour
            )

            aspect_ratio = float(w) / max(h, 1)

            # Analysis Region Gate (45% to 82% screen width = ~1.5m to 2.5m directly in front)
            MIN_BOARD_WIDTH = int(
                img_w * 0.45
            )

            MAX_BOARD_WIDTH = int(
                img_w * 0.82
            )

            if (
                MIN_BOARD_WIDTH < w < MAX_BOARD_WIDTH
                and y > 8
                and aspect_ratio > 1.8
            ):
                board_crop = image[
                    y:y+h,
                    x:x+w
                ]

                # Dynamic Slicing: 6-panel or 3-panel board
                if aspect_ratio >= 3.5:
                    layout = BOARD_LAYOUT
                    num_panels = 6

                else:
                    layout = (
                        ['A', 'B', 'C']
                        if x < (img_w / 2)
                        else ['X', 'Y', 'Z']
                    )

                    num_panels = 3

                panel_w = w // num_panels

                for i in range(num_panels):
                    letter = layout[i]

                    panel = board_crop[
                        :,
                        i*panel_w:(i+1)*panel_w
                    ]

                    if (
                        panel.shape[0] < 10
                        or panel.shape[1] < 10
                    ):
                        continue

                    # Strict Arrow Isolation (Bottom 42% of panel - 0% letter legs)
                    mid_y = int(
                        panel.shape[0] * 0.58
                    )

                    arrow_crop = panel[
                        mid_y:,
                        :
                    ]

                    direction = self.classify_arrow_crop(
                        arrow_crop
                    )

                    if direction is not None:
                        # 5-Frame Confirmation Analysis
                        if (
                            self.confirm_counts[letter]['dir']
                            == direction
                        ):
                            self.confirm_counts[letter]['count'] += 1

                        else:
                            self.confirm_counts[letter]['dir'] = direction
                            self.confirm_counts[letter]['count'] = 1

                        if (
                            self.confirm_counts[letter]['count']
                            >= self.CONFIRM_THRESHOLD
                        ):
                            msg = String()
                            msg.data = (
                                f"{letter}_{direction}"
                            )

                            self.publisher_sign.publish(
                                msg
                            )

                            self.confirm_counts[letter]['count'] = 0

                            self.get_logger().info(
                                f"📡 CONFIRMED SIGN: {letter}_{direction}"
                            )


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