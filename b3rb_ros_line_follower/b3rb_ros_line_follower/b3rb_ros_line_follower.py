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
        self.create_subscription(
            EdgeVectors,
            '/edge_vectors',
            self.edge_vectors_callback,
            QOS_PROFILE_DEFAULT
        )
        self.create_subscription(
            LaserScan,
            '/scan',
            self.lidar_callback,
            QOS_PROFILE_DEFAULT
        )
        self.create_subscription(
            ServerCommunication,
            '/ServerCommunication',
            self.server_communication_callback,
            QOS_PROFILE_DEFAULT
        )
        self.create_subscription(
            String,
            '/qr_detection',
            self.qr_detection_callback,
            QOS_PROFILE_DEFAULT
        )
        self.create_subscription(
            String,
            '/sign_board_detection',
            self.sign_board_callback,
            QOS_PROFILE_DEFAULT
        )

        self.publisher_joy = self.create_publisher(
            Joy, '/cerebri/in/joy', QOS_PROFILE_DEFAULT
        )
        self.publisher_server = self.create_publisher(
            ServerCommunication, '/ServerCommunication', QOS_PROFILE_DEFAULT
        )

        # Mission State
        self.mission_state = "SEARCH_PATIENT"
        self.patient_sequence = ['A', 'B', 'C']
        self.patient_index = 0
        self.target_letter = self.patient_sequence[0]
        self.expected_hospital = None
        self.patients_delivered = 0
        self.outbound_uid = 100
        self.parking_timer_start = 0.0

        # Steering, Speed, & Ramping (EXTREME HIGH SPEED SETTINGS)
        self.MAX_SPEED = 10.0         # Massive straightaway speed!
        self.MIN_CORNER_SPEED = 1.80  # Much faster cornering
        self.target_speed = 1.50      
        self.current_actual_speed = 0.0
        self.target_turn = 0.0
        self.prev_error = 0.0
        self.last_turn_direction = 0.0

        self.active_sign_command = None
        self.sign_command_timestamp = 0.0

        # LIDAR Shield & Stopping
        self.MIN_VALID_LIDAR_DIST = 0.30
        self.ZONE_RANGE_MAX = 1.20
        self.sensed_qr = None
        self.zone_stop_timer_start = 0.0
        
        # Stop Delay for Glide
        self.ZONE_STOP_DELAY = 0.5    
        
        self.front_building_range = 10.0
        self.obstacle_in_front = False
        self.obstacle_target_shift = 0.0

        self.FRONT_STOP_DIST = 0.20

        # Smart Recovery
        self.driving_state = "FORWARD"
        self.recovery_stage_start = 0.0
        self.close_front_count = 0
        self.trap_location = "CENTER"
        self.FRONT_TRAP_DIST = 0.28
        self.TRAP_DEBOUNCE_COUNT = 3
        self.min_rear = 10.0

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
            self.current_actual_speed = 0.0
            msg.axes = [0.0, 0.0, 0.0, 0.0]
            self.publisher_joy.publish(msg)
            return

        if self.driving_state == "RECOVERY":
            self.current_actual_speed = 0.0
            elapsed = time.time() - self.recovery_stage_start

            if elapsed > 2.0 or self.min_rear < 0.25:
                if not self.obstacle_in_front:
                    self.driving_state = "FORWARD"
                    self.recovery_stage_start = 0.0
                else:
                    self.recovery_stage_start = time.time()
                return

            escape_turn = 0.8 if self.trap_location == "LEFT" else -0.8 if self.trap_location == "RIGHT" else 0.0
            msg.axes = [0.0, -0.35, 0.0, escape_turn]
            self.publisher_joy.publish(msg)
            return

        desired_speed = 0.0 if self.obstacle_in_front else self.target_speed

        # AGGRESSIVE ACCELERATION & BRAKING FOR SPEED 10
        if self.current_actual_speed < desired_speed:
            self.current_actual_speed += 0.30  # Very fast acceleration
            self.current_actual_speed = min(self.current_actual_speed, desired_speed)
        elif self.current_actual_speed > desired_speed:
            # Massive braking force needed to slow down from 10 m/s
            brake_force = 1.50 if desired_speed == 0.0 else 0.80
            self.current_actual_speed -= brake_force
            self.current_actual_speed = max(self.current_actual_speed, desired_speed)

        self.current_actual_speed = float(np.clip(self.current_actual_speed, 0.0, self.MAX_SPEED))
        final_turn = float(np.clip(self.target_turn, TURN_MIN, TURN_MAX))

        msg.axes = [0.0, self.current_actual_speed, 0.0, final_turn]
        self.publisher_joy.publish(msg)

    def get_sector_ranges(self, message, center_deg, half_width_deg):
        angle_min, angle_increment = message.angle_min, message.angle_increment
        n = len(message.ranges)
        if n == 0 or angle_increment == 0: return []
        def to_index(deg): return int(round((math.radians(deg) - angle_min) / angle_increment)) % n
        i0, i1 = to_index(center_deg - half_width_deg), to_index(center_deg + half_width_deg)
        sector = message.ranges[i0:i1 + 1] if i0 <= i1 else list(message.ranges[i0:]) + list(message.ranges[:i1 + 1])
        return [r for r in sector if self.MIN_VALID_LIDAR_DIST < r < message.range_max]

    def lidar_callback(self, message):
        front_center = self.get_sector_ranges(message, 0, 15)
        front_left   = self.get_sector_ranges(message, 25, 15)   
        front_right  = self.get_sector_ranges(message, -25, 15)  

        min_fc = min(front_center) if front_center else 10.0
        min_fl = min(front_left) if front_left else 10.0
        min_fr = min(front_right) if front_right else 10.0

        self.zone_check_range = min(min_fc, min_fl, min_fr)
        self.obstacle_in_front = min_fc < self.FRONT_STOP_DIST

        AVOID_DIST = 1.15
        MAX_SHIFT = 60.0

        force_left = max(0.0, 1.0 - (min_fl / AVOID_DIST))
        force_right = max(0.0, 1.0 - (min_fr / AVOID_DIST))

        raw_shift = (force_left * MAX_SHIFT) - (force_right * MAX_SHIFT)
        raw_shift = float(np.clip(raw_shift, -MAX_SHIFT, MAX_SHIFT))

        if abs(raw_shift) < 5.0:
            raw_shift = 0.0

        if abs(raw_shift) > abs(self.obstacle_target_shift):
            self.obstacle_target_shift = raw_shift
        else:
            self.obstacle_target_shift = (0.85 * self.obstacle_target_shift) + (0.15 * raw_shift)

        self.check_zone_arrival()

    def edge_vectors_callback(self, message):
        if self.mission_state in HOLD_STATES:
            return

        width = message.image_width
        height = message.image_height

        if width == 0 or height == 0:
            return

        car_x = width / 2.0
        car_y = float(height)

        # 1. TIMEOUT REDUCED TO 5 SECONDS (Was 16s)
        if self.active_sign_command is not None and getattr(self, 'sign_command_timestamp', 0.0) > 0.0:
            if (time.time() - self.sign_command_timestamp) > 5.0:
                self.active_sign_command = None
                self.in_intersection = False
                self.sign_command_timestamp = 0.0
                self.get_logger().info("⏱️ Turn completed (5s timeout). Memory automatically reset.")

        LANE_HALF_WIDTH = width * 0.30
        SINGLE_LINE_OFFSET = width * 0.42

        has_active_sign = self.active_sign_command is not None

        v1 = message.vector_1 if message.vector_count > 0 else None
        v2 = message.vector_2 if message.vector_count > 1 else None

        if message.vector_count == 2 and v1[0].x > v2[0].x:
            v1, v2 = v2, v1

        is_fork = False

        if message.vector_count == 2:
            lane_width_top = abs(v2[0].x - v1[0].x)

            if lane_width_top > (width * 0.38):
                is_fork = True

            elif lane_width_top < (width * 0.35):
                if getattr(self, 'in_intersection', False):
                    self.active_sign_command = None
                    self.in_intersection = False
                    self.sign_command_timestamp = 0.0
                    self.get_logger().info("🛣️ Lane safely realigned. Memory wiped for next signboard.")

        if is_fork and has_active_sign:
            self.in_intersection = True

        active_count = message.vector_count

        if is_fork and has_active_sign:
            if "LEFT" in self.active_sign_command:
                v2 = None
                active_count = 1
            elif "RIGHT" in self.active_sign_command or "STRAIGHT" in self.active_sign_command:
                v1 = v2
                v2 = None
                active_count = 1

        # TARGET CALCULATION
        if active_count == 2:
            target_x = (v1[0].x + v2[0].x) / 2.0
            target_y = (v1[0].y + v2[0].y) / 2.0

        elif active_count == 1:
            top_x, top_y = v1[0].x, v1[0].y
            bot_x = v1[1].x
            avg_x = (top_x + bot_x) / 2.0

            if self.active_sign_command == "LEFT":
                if avg_x < car_x + 50:
                    target_x, target_y = top_x + SINGLE_LINE_OFFSET, top_y
                else:   
                    target_x, target_y = car_x - 120.0, car_y - 70.0

            elif self.active_sign_command == "RIGHT" or self.active_sign_command == "STRAIGHT":
                if avg_x > car_x - 50:
                    target_x, target_y = top_x - SINGLE_LINE_OFFSET, top_y
                else:
                    target_x, target_y = car_x + 120.0, car_y - 70.0

            else:
                if avg_x < car_x:
                    target_x, target_y = top_x + SINGLE_LINE_OFFSET, top_y
                else:
                    target_x, target_y = top_x - SINGLE_LINE_OFFSET, top_y

        else:
            if self.active_sign_command == "LEFT":
                target_x, target_y = car_x - 120.0, car_y - 70.0
            else:
                target_x, target_y = car_x + 120.0, car_y - 70.0

        safe_shift = np.clip(self.obstacle_target_shift, -80.0, 80.0)
        target_x += safe_shift
        target_x = np.clip(target_x, 20.0, width - 20.0)

        dx = car_x - target_x
        dy = car_y - target_y

        # DYNAMIC LOOKAHEAD CAPPED (Crucial for Speed 10!)
        dynamic_lookahead = 70.0 + (self.current_actual_speed * 30.0)
        dynamic_lookahead = min(dynamic_lookahead, height * 0.85) # Prevents looking off-screen
        if dy < dynamic_lookahead:
            dy = dynamic_lookahead

        theta_error = math.atan2(dx, dy)
        derivative = theta_error - self.prev_error
        self.prev_error = theta_error

        kp_angle = 0.75; kd_angle = 0.40
        raw_turn = (kp_angle * theta_error) + (kd_angle * derivative)

        smooth_turn = (0.65 * self.target_turn) + (0.35 * raw_turn)
        self.last_turn_direction = float(np.clip(smooth_turn, -1.0, 1.0))

        # --- PRE-EMPTIVE BRAKING LOGIC ---
        approaching_target_qr = False
        if self.sensed_qr:
            if self.mission_state == "SEARCH_PATIENT" and self.sensed_qr == SIGN_TO_PATIENT.get(self.target_letter):
                approaching_target_qr = True
            elif self.mission_state == "SEARCH_HOSPITAL" and self.sensed_qr == self.expected_hospital:
                approaching_target_qr = True

        # --- EXTREME SPEED MODULATION (Highest Priority First) ---
        if approaching_target_qr:
            # #1 PRIORITY: MUST be slow to stop accurately in the zone! Override all high speeds.
            self.target_speed = 0.50   
        elif abs(theta_error) > 0.35:
            self.target_speed = 1.80   # Sharp corners (Increased)
        elif abs(theta_error) > 0.15:
            self.target_speed = 3.50   # Mild corners (Increased)
        elif self.active_sign_command is not None:
            self.target_speed = 4.00   # Fast passing speed for signboards (Increased)
        else:
            self.target_speed = self.MAX_SPEED # SPEED 10.0 🚀

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

        self.active_sign_command = None
        self.in_intersection = False
        self.sign_command_timestamp = 0.0

        if self.mission_state == "SEARCH_PATIENT":
            resolved_patient = self.resolve_patient_payload(payload)
            if resolved_patient:
                self.target_letter = PATIENT_TO_SIGN[resolved_patient]
                self.get_logger().info(f"🔄 SERVER DIRECTIVE: Target Patient set to {resolved_patient} (Route Sign: {self.target_letter})")

        elif self.mission_state == "AT_PATIENT_ZONE_WAIT":
            resolved_hospital = self.resolve_hospital_payload(payload)
            if resolved_hospital:
                self.expected_hospital = resolved_hospital
                self.target_letter = HOSPITAL_TO_SIGN[resolved_hospital]
                self.mission_state = "SEARCH_HOSPITAL"
                self.get_logger().info(f"✅ ASSIGNED HOSPITAL: {resolved_hospital} (Route Sign: {self.target_letter})")

        elif self.mission_state == "AT_HOSPITAL_ZONE_WAIT":
            self.patients_delivered += 1
            if self.patients_delivered >= 3:
                self.mission_state = "EXIT_TO_PARK"
                self.get_logger().info("✅ ALL PATIENTS DELIVERED: Proceeding to Park.")
            else:
                self.patient_index += 1
                resolved_patient = self.resolve_patient_payload(payload)
                self.target_letter = PATIENT_TO_SIGN[resolved_patient] if resolved_patient else self.patient_sequence[self.patient_index]
                self.mission_state = "SEARCH_PATIENT"
                self.get_logger().info(f"✅ NEXT PATIENT: Proceed to {self.target_letter}")

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

    def check_zone_arrival(self):
        if not self.sensed_qr: return

        zone_confirmed = self.zone_check_range <= self.ZONE_RANGE_MAX

        if self.mission_state == "SEARCH_PATIENT":
            expected = SIGN_TO_PATIENT.get(self.target_letter)
            if self.sensed_qr == expected and zone_confirmed:
                if self.zone_stop_timer_start == 0.0:
                    self.zone_stop_timer_start = time.time()
                    self.get_logger().info(f"🎯 ENTERING PATIENT ZONE for {expected}: Rolling forward into center...")

                elif (time.time() - self.zone_stop_timer_start) >= self.ZONE_STOP_DELAY:
                    self.send_server_update(self.sensed_qr)
                    self.mission_state = "AT_PATIENT_ZONE_WAIT"
                    self.sensed_qr = None
                    self.active_sign_command = None
                    self.in_intersection = False
                    self.sign_command_timestamp = 0.0
                    self.zone_stop_timer_start = 0.0
                    self.get_logger().info(f"🛑 CENTERED IN PATIENT ZONE: Stopped for {expected}")

        elif self.mission_state == "SEARCH_HOSPITAL":
            if self.sensed_qr == self.expected_hospital and zone_confirmed:
                if self.zone_stop_timer_start == 0.0:
                    self.zone_stop_timer_start = time.time()
                    self.get_logger().info(f"🎯 ENTERING HOSPITAL ZONE for {self.expected_hospital}: Rolling forward into center...")

                elif (time.time() - self.zone_stop_timer_start) >= self.ZONE_STOP_DELAY:
                    self.send_server_update(self.sensed_qr)
                    self.mission_state = "AT_HOSPITAL_ZONE_WAIT"
                    self.sensed_qr = None
                    self.active_sign_command = None
                    self.in_intersection = False
                    self.sign_command_timestamp = 0.0
                    self.zone_stop_timer_start = 0.0
                    self.get_logger().info(f"🛑 CENTERED IN HOSPITAL ZONE: Stopped for {self.expected_hospital}")

    def qr_detection_callback(self, message):
        qr_data = self.normalize_qr_payload(message.data)
        if qr_data in FAKE_HOSPITALS: return
        self.sensed_qr = qr_data
        self.check_zone_arrival()

    def sign_board_callback(self, message):
        parts = message.data.upper().strip().split('_', 1)
        if len(parts) != 2: 
            return
        
        letter, direction = parts

        if letter != self.target_letter: 
            return

        if getattr(self, 'in_intersection', False) and self.active_sign_command is not None:
            return

        if self.active_sign_command != direction:
            self.active_sign_command = direction
            self.sign_command_timestamp = time.time()
            self.get_logger().warn(f"🎯 MATCHED TARGET SIGN ({letter})! Intent CONFIRMED & LOCKED: {direction}")


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