# Copyright 2024-2026 NXP
# Licensed under the Apache License, Version 2.0 

import rclpy
from rclpy.node import Node
import time
import math
import re
import numpy as np
from sensor_msgs.msg import Joy, LaserScan
from std_msgs.msg import String
from synapse_msgs.msg import EdgeVectors, ServerCommunication

QOS_PROFILE_DEFAULT = 10
TURN_MIN = -1.0
TURN_MAX = 1.0

# Mission Protocol
SIGN_TO_PATIENT = {'A': 'PATIENT_1', 'B': 'PATIENT_2', 'C': 'PATIENT_3'}
SIGN_TO_HOSPITAL = {'X': 'HOSPITAL_1', 'Y': 'HOSPITAL_2', 'Z': 'HOSPITAL_3'}
PATIENT_TO_SIGN = {v: k for k, v in SIGN_TO_PATIENT.items()}
HOSPITAL_TO_SIGN = {v: k for k, v in SIGN_TO_HOSPITAL.items()}
FAKE_HOSPITALS = {'FAKE_HOSPITAL_1', 'FAKE_HOSPITAL_2'}
HOLD_STATES = {"AT_PATIENT_ZONE_WAIT", "AT_HOSPITAL_ZONE_WAIT", "PARKED_WAIT_ACK", "DONE"}

class LineFollower(Node):
    def __init__(self):
        super().__init__('line_follower')

        # Subscriptions & Publishers
        self.create_subscription(EdgeVectors, '/edge_vectors', self.edge_vectors_callback, QOS_PROFILE_DEFAULT)
        self.create_subscription(LaserScan, '/scan', self.lidar_callback, QOS_PROFILE_DEFAULT)
        self.create_subscription(ServerCommunication, '/ServerCommunication', self.server_communication_callback, QOS_PROFILE_DEFAULT)
        self.create_subscription(String, '/qr_detection', self.qr_detection_callback, QOS_PROFILE_DEFAULT)
        self.create_subscription(String, '/sign_board_detection', self.sign_board_callback, QOS_PROFILE_DEFAULT)

        self.publisher_joy = self.create_publisher(Joy, '/cerebri/in/joy', QOS_PROFILE_DEFAULT)
        self.publisher_server = self.create_publisher(ServerCommunication, '/ServerCommunication', QOS_PROFILE_DEFAULT)

        # Mission State
        self.mission_state = "SEARCH_PATIENT"
        self.patient_sequence = ['A', 'B', 'C']
        self.patient_index = 0
        self.target_letter = self.patient_sequence[0]
        self.expected_hospital = None
        self.patients_delivered = 0
        self.outbound_uid = 100
        self.parking_timer_start = 0.0

        # Steering & Drive (Pure Pursuit)
        self.target_speed = 0.38
        self.target_turn = 0.0
        self.prev_error = 0.0
        self.last_turn_direction = 0.0
        self.active_sign_command = None
        self.sign_command_timestamp = 0.0

        # Smart Recovery & LIDAR Shield
        self.driving_state = "FORWARD"
        self.recovery_stage_start = 0.0
        self.close_front_count = 0
        self.trap_location = "CENTER"
        
        self.FRONT_SLOW_DIST = 1.3
        self.FRONT_STOP_DIST = 0.45
        self.FRONT_TRAP_DIST = 0.28
        self.TRAP_DEBOUNCE_COUNT = 3
        self.MIN_VALID_LIDAR_DIST = 0.26  # Chassis filter
        self.ZONE_RANGE_MAX = 0.65

        self.min_rear = 10.0
        self.front_building_range = 10.0
        self.obstacle_in_front = False
        self.obstacle_target_shift = 0.0

        self.control_timer = self.create_timer(0.1, self.publish_drive_commands)

    def check_if_parked(self):
        if self.parking_timer_start == 0.0:
            self.parking_timer_start = time.time()
            return False
        if (time.time() - self.parking_timer_start) > 12.0:
            return True
        return False

    def publish_drive_commands(self):
        msg = Joy()
        msg.buttons = [1, 0, 0, 0, 0, 0, 0, 1]

        if self.mission_state == "EXIT_TO_PARK" and self.check_if_parked():
            self.send_server_update("PARKED")
            self.mission_state = "PARKED_WAIT_ACK"

        if self.mission_state in HOLD_STATES:
            msg.axes = [0.0, 0.0, 0.0, 0.0]
            self.publisher_joy.publish(msg)
            return

        # ---------------------------------------------------------
        # SPATIALLY-AWARE ACKERMANN RECOVERY
        # ---------------------------------------------------------
        if self.driving_state == "RECOVERY":
            elapsed = time.time() - self.recovery_stage_start
            
            # Stop reversing after 2 seconds or if rear is blocked
            if elapsed > 2.0 or self.min_rear < 0.25:
                self.driving_state = "FORWARD"
                self.recovery_stage_start = 0.0
                return

            # Smart Steering: Steer TOWARDS the side we are trapped on while in reverse.
            # This causes the nose of the buggy to violently swing AWAY from the obstacle!
            escape_turn = 0.0
            if self.trap_location == "LEFT":
                escape_turn = 0.8   # Steer Left -> Nose swings Right
            elif self.trap_location == "RIGHT":
                escape_turn = -0.8  # Steer Right -> Nose swings Left
                
            msg.axes = [0.0, -0.35, 0.0, escape_turn]
            self.publisher_joy.publish(msg)
            return

        # ---------------------------------------------------------
        # NORMAL DRIVING
        # ---------------------------------------------------------
        current_speed = 0.0 if self.obstacle_in_front else self.target_speed
        msg.axes = [0.0, current_speed, 0.0, self.target_turn]
        self.publisher_joy.publish(msg)

    def get_sector_ranges(self, message, center_deg, half_width_deg):
        angle_min, angle_increment = message.angle_min, message.angle_increment
        n = len(message.ranges)
        if n == 0 or angle_increment == 0: return []
        def to_index(deg): return int(round((math.radians(deg) - angle_min) / angle_increment)) % n
        i0, i1 = to_index(center_deg - half_width_deg), to_index(center_deg + half_width_deg)
        sector = message.ranges[i0:i1+1] if i0 <= i1 else list(message.ranges[i0:]) + list(message.ranges[:i1+1])
        return [r for r in sector if self.MIN_VALID_LIDAR_DIST < r < message.range_max]

    def lidar_callback(self, message):
        """180-Degree Trap Detection & Dynamic Safety Shield"""
        front_center = self.get_sector_ranges(message, 0, 15)
        front_left   = self.get_sector_ranges(message, 45, 30)
        front_right  = self.get_sector_ranges(message, -45, 30)
        rear         = self.get_sector_ranges(message, 180, 30)

        min_fc = min(front_center) if front_center else 10.0
        min_fl = min(front_left) if front_left else 10.0
        min_fr = min(front_right) if front_right else 10.0
        self.min_rear = min(rear) if rear else 10.0
        self.front_building_range = min(min_fc, min_fl, min_fr)

        # 1. TRAP DETECTION (Checks all 3 front sectors!)
        # If any part of the front bumper or side fenders hits something, trigger recovery.
        trap_dist = min(min_fc, min_fl, min_fr)
        
        if trap_dist < self.FRONT_TRAP_DIST:
            self.close_front_count += 1
            # Log exactly where we got trapped to inform the Ackermann recovery
            if min_fl == trap_dist: self.trap_location = "LEFT"
            elif min_fr == trap_dist: self.trap_location = "RIGHT"
            else: self.trap_location = "CENTER"
        else:
            self.close_front_count = 0

        if self.driving_state == "FORWARD" and self.close_front_count >= self.TRAP_DEBOUNCE_COUNT:
            self.get_logger().warn(f"Trapped on {self.trap_location}! Initiating Smart Recovery.")
            self.driving_state = "RECOVERY"
            self.recovery_stage_start = time.time()
            self.close_front_count = 0
            return

        self.obstacle_in_front = min_fc < self.FRONT_STOP_DIST
        
        # 2. VIRTUAL SAFETY SHIELD
        AVOID_DIST = 1.3
        MAX_SHIFT = 140.0
        raw_shift = 0.0

        if min_fl < AVOID_DIST or min_fr < AVOID_DIST:
            if min_fl < min_fr:
                intensity = 1.0 - (min_fl / AVOID_DIST)
                raw_shift = intensity * MAX_SHIFT
            else:
                intensity = 1.0 - (min_fr / AVOID_DIST)
                raw_shift = -intensity * MAX_SHIFT

        # Smooth Evasion Decay (Sticky Memory)
        if abs(raw_shift) > abs(self.obstacle_target_shift):
            self.obstacle_target_shift = raw_shift
        else:
            self.obstacle_target_shift = (0.85 * self.obstacle_target_shift) + (0.15 * raw_shift)

    def edge_vectors_callback(self, message):
        """Kinematic Pure Pursuit combined with the Safety Shield Offset."""
        if self.driving_state == "RECOVERY" or self.mission_state in HOLD_STATES:
            return

        width = message.image_width
        height = message.image_height
        if width == 0 or height == 0: 
            return

        car_x = width / 2.0
        car_y = float(height)
        LANE_HALF_WIDTH = width * 0.30 
        SINGLE_LINE_OFFSET = width * 0.42  

        has_active_sign = (self.active_sign_command is not None) and ((time.time() - self.sign_command_timestamp) < 2.0)

        if message.vector_count == 2:
            l_top_x, l_top_y = message.vector_1[0].x, message.vector_1[0].y
            r_top_x, r_top_y = message.vector_2[0].x, message.vector_2[0].y
            lane_width_top = abs(r_top_x - l_top_x)

            if lane_width_top > (width * 0.45):
                if has_active_sign and "LEFT" in self.active_sign_command:
                    target_x, target_y = l_top_x + LANE_HALF_WIDTH, l_top_y
                elif has_active_sign and "RIGHT" in self.active_sign_command:
                    target_x, target_y = r_top_x - LANE_HALF_WIDTH, r_top_y
                else:
                    target_x, target_y = r_top_x - LANE_HALF_WIDTH, r_top_y
            else:
                target_x = (l_top_x + r_top_x) / 2.0
                target_y = (l_top_y + r_top_y) / 2.0

        elif message.vector_count == 1:
            top_x, top_y = message.vector_1[0].x, message.vector_1[0].y
            bot_x = message.vector_1[1].x  
            avg_x = (top_x + bot_x) / 2.0

            if avg_x < car_x:
                target_x, target_y = top_x + SINGLE_LINE_OFFSET, top_y
            else:
                target_x, target_y = top_x - SINGLE_LINE_OFFSET, top_y
        else:
            if abs(self.last_turn_direction) > 0.1:
                self.target_turn = self.last_turn_direction * 0.85
                self.target_speed = 0.16
                return
            else:
                target_x, target_y = car_x, car_y - 70.0

        # APPLY SAFETY SHIELD TO PURE PURSUIT TARGET!
        target_x += self.obstacle_target_shift

        dx = car_x - target_x  
        dy = car_y - target_y
        if dy < 70.0: dy = 70.0 

        theta_error = math.atan2(dx, dy)
        derivative = theta_error - self.prev_error
        self.prev_error = theta_error

        kp_angle = 0.75
        kd_angle = 0.40
        raw_turn = (kp_angle * theta_error) + (kd_angle * derivative)

        smooth_turn = (0.70 * getattr(self, 'target_turn', 0.0)) + (0.30 * raw_turn)
        self.last_turn_direction = float(np.clip(smooth_turn, -1.0, 1.0))

        if abs(theta_error) > 0.35:    
            self.target_speed = 0.14 
        elif abs(theta_error) > 0.15:  
            self.target_speed = 0.22
        else:                          
            self.target_speed = 0.35

        self.target_turn = self.last_turn_direction

    def resolve_hospital_payload(self, payload):
        if payload in SIGN_TO_HOSPITAL: return SIGN_TO_HOSPITAL[payload]
        if payload in SIGN_TO_HOSPITAL.values(): return payload
        return None

    def resolve_patient_payload(self, payload):
        if payload in SIGN_TO_PATIENT: return SIGN_TO_PATIENT[payload]
        if payload in SIGN_TO_PATIENT.values(): return payload
        return None

    def server_communication_callback(self, message):
        if message.dest != 1: return
        ack = ServerCommunication()
        ack.src, ack.dest, ack.uid, ack.ack, ack.msg = 1, 2, message.uid, 1, ""
        self.publisher_server.publish(ack)

        payload = message.msg.strip().upper()
        if not payload: return

        if self.mission_state == "AT_PATIENT_ZONE_WAIT":
            resolved = self.resolve_hospital_payload(payload)
            if resolved:
                self.expected_hospital = resolved
                self.target_letter = HOSPITAL_TO_SIGN[resolved]
                self.mission_state = "SEARCH_HOSPITAL"
        elif self.mission_state == "AT_HOSPITAL_ZONE_WAIT":
            self.patients_delivered += 1
            if self.patients_delivered >= 3:
                self.mission_state = "EXIT_TO_PARK"
            else:
                self.patient_index += 1
                resolved = self.resolve_patient_payload(payload)
                self.target_letter = PATIENT_TO_SIGN[resolved] if resolved else self.patient_sequence[self.patient_index]
                self.mission_state = "SEARCH_PATIENT"
        elif self.mission_state == "PARKED_WAIT_ACK":
            if payload == "OK": self.mission_state = "DONE"

    def send_server_update(self, text_msg):
        m = ServerCommunication()
        m.src, m.dest, m.uid, m.ack, m.msg = 1, 2, self.outbound_uid, 0, text_msg
        self.publisher_server.publish(m)
        self.outbound_uid = (self.outbound_uid + 1) % 256

    def normalize_qr_payload(self, raw):
        match = re.search(r'\{?\s*LOC:\s*([A-Z_0-9]+)\s*\}?', raw)
        return match.group(1) if match else raw.strip().upper()

    def qr_detection_callback(self, message):
        qr_data = self.normalize_qr_payload(message.data)
        zone_confirmed = self.front_building_range < self.ZONE_RANGE_MAX

        if self.mission_state == "SEARCH_PATIENT":
            expected = SIGN_TO_PATIENT.get(self.target_letter)
            if qr_data == expected and zone_confirmed:
                self.send_server_update(qr_data)
                self.mission_state = "AT_PATIENT_ZONE_WAIT"
        elif self.mission_state == "SEARCH_HOSPITAL":
            if qr_data in FAKE_HOSPITALS: return
            if qr_data == self.expected_hospital and zone_confirmed:
                self.send_server_update(qr_data)
                self.mission_state = "AT_HOSPITAL_ZONE_WAIT"

    def sign_board_callback(self, message):
        parts = message.data.upper().strip().split('_', 1)
        if len(parts) != 2: return
        letter, direction = parts
        if letter != self.target_letter: return
        self.active_sign_command = direction
        self.sign_command_timestamp = time.time()

def main(args=None):
    rclpy.init(args=args)
    node = LineFollower()
    try: rclpy.spin(node)
    except KeyboardInterrupt: pass
    finally:
        node.destroy_node()
        if rclpy.ok(): rclpy.shutdown()

if __name__ == '__main__':
    main()