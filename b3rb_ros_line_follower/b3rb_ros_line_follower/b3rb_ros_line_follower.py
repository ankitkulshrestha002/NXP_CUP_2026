# Copyright 2024-2026 NXP
# Licensed under the Apache License, Version 2.0 

import rclpy
from rclpy.node import Node
import time
import math
import re
import numpy as np
from sensor_msgs.msg import Joy, LaserScan
from nav_msgs.msg import Odometry
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
        self.create_subscription(
            String,
            '/cone_detection_cam',
            self.cone_detection_callback,
            QOS_PROFILE_DEFAULT
        )
        self.create_subscription(
            Odometry,
            '/cerebri/out/odometry',
            self.odometry_callback,
            QOS_PROFILE_DEFAULT
        )

        self.publisher_joy = self.create_publisher(
            Joy, '/cerebri/in/joy', QOS_PROFILE_DEFAULT
        )
        self.publisher_server = self.create_publisher(
            ServerCommunication, '/ServerCommunication', QOS_PROFILE_DEFAULT
        )
        self.publisher_mission_state = self.create_publisher(
            String, '/mission_state', QOS_PROFILE_DEFAULT
        )

        # Mission State
        self.mission_state = "SEARCH_PATIENT"
        self.patient_sequence = ['A', 'B', 'C']
        self.patient_index = 0
        self.target_letter = self.patient_sequence[0]
        self.expected_hospital = None
        self.patients_delivered = 0
        self.outbound_uid = 100

        # Parking State Machine (Kinematic Reverse Swing)
        self.parking_phase = 0          # 0: SEARCH, 1: CLEAR, 2: SWING, 3: STRAIGHT, 4: DONE
        self.parking_cone_side = None    # "LEFT" or "RIGHT"
        self.parking_start_time = 0.0    # For LIDAR fallback timeout
        self.parking_timer_start = 0.0   # For swing sequence timing
        
        # Universal Smart Parking Alignment Variables
        self.cone_aligned = False
        self.clearance_timer_start = 0.0
        self.side_left = 10.0
        self.side_right = 10.0

        # Odometry tracking for distance-based cone suppression
        self.odom_x = 0.0
        self.odom_y = 0.0
        self.parking_start_x = 0.0
        self.parking_start_y = 0.0
        
        self.CONE_DETECT_MIN_DIST = 1.5  

        # Steering, Speed, & Ramping 
        self.MAX_SPEED = 2.0        
        self.MIN_CORNER_SPEED = 0.50
        self.target_speed = 5.00      
        self.current_actual_speed = 0.0
        self.target_turn = 0.0
        self.prev_error = 0.0
        self.last_turn_direction = 0.0

        self.active_sign_command = None
        self.sign_command_timestamp = 0.0

        # STRAIGHT phase machine
        self.straight_phase = None          
        self.straight_zero_start = 0.0      
        self.straight_follow_start = 0.0    

        # Sign confirmation
        self.last_received_sign = None
        self.sign_confirm_count = 0

        # LIDAR Shield & Stopping
        self.MIN_VALID_LIDAR_DIST = 0.30
        self.ZONE_RANGE_MAX = 1.20
        self.sensed_qr = None
        self.zone_stop_timer_start = 0.0
        
        self.ZONE_STOP_DELAY = 2.5   
        
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

    def set_mission_state(self, new_state):
        self.mission_state = new_state
        msg = String()
        msg.data = new_state
        self.publisher_mission_state.publish(msg)
        self.get_logger().info(f"📋 Mission state → {new_state}")

    def publish_drive_commands(self):
        msg = Joy()
        msg.buttons = [1, 0, 0, 0, 0, 0, 0, 1]

        # =========================================================
        # KINEMATIC REVERSE SWING PARKING LOGIC (UNIVERSAL SMART ALIGNMENT)
        # =========================================================
        if self.mission_state == "EXIT_TO_PARK":
            elapsed = time.time() - self.parking_timer_start

            if self.parking_phase == 0:
                self.target_speed = 0.40 

            elif self.parking_phase == 1:
                self.current_actual_speed = 0.60   
                self.target_turn = 0.0
                
                side_dist = self.side_left if self.parking_cone_side == "LEFT" else self.side_right

                if side_dist < 1.5 and not self.cone_aligned:
                    self.cone_aligned = True
                    self.clearance_timer_start = time.time()
                    self.get_logger().info(f"🅿️ Cones perfectly beside us! Clearing rear bumper...")

                if self.cone_aligned:
                    if (time.time() - self.clearance_timer_start) > 2.4:  
                        self.parking_phase = 2
                        self.parking_timer_start = time.time()
                        self.get_logger().info(f"🅿️ Swinging reverse into {self.parking_cone_side}.")
                else:
                    if elapsed > 6.0:
                        self.parking_phase = 2
                        self.parking_timer_start = time.time()
                        self.get_logger().warn("🅿️ Side LiDAR missed cones. Forcing swing fallback.")
            
            elif self.parking_phase == 2:
                self.current_actual_speed = -0.40  
                self.target_turn = 1.0 if self.parking_cone_side == "LEFT" else -1.0
                
                if elapsed > 3.2:  
                    self.parking_phase = 3
                    self.parking_timer_start = time.time()
                    self.get_logger().info("🅿️ Straightening out...")

            elif self.parking_phase == 3:
                self.current_actual_speed = -0.40  
                self.target_turn = 0.0
                
                if elapsed > 3.0 or self.min_rear < 0.45:  
                    self.parking_phase = 4
                    self.get_logger().info(f"🅿️ Successfully Parked! (Rear Dist: {self.min_rear:.2f}m)")

            elif self.parking_phase == 4:
                self.current_actual_speed = 0.0
                self.target_turn = 0.0
                self.send_server_update("PARKED")
                self.set_mission_state("PARKED_WAIT_ACK")
                return

            if self.parking_phase in [1, 2, 3]:
                msg.axes = [0.0, self.current_actual_speed, 0.0, float(self.target_turn)]
                self.publisher_joy.publish(msg)
                return

        if self.mission_state in HOLD_STATES:
            if self.current_actual_speed > 0.0:
                self.current_actual_speed -= 1.80  # SLAM ON BRAKES
                self.current_actual_speed = max(0.0, self.current_actual_speed)
            msg.axes = [0.0, self.current_actual_speed, 0.0, 0.0]
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

        if self.current_actual_speed < desired_speed:
            self.current_actual_speed += 0.08  
            self.current_actual_speed = min(self.current_actual_speed, desired_speed)
        elif self.current_actual_speed > desired_speed:
            brake_force = 0.40 if desired_speed == 0.0 else 0.20
            self.current_actual_speed -= brake_force
            self.current_actual_speed = max(self.current_actual_speed, desired_speed)

        self.current_actual_speed = float(np.clip(self.current_actual_speed, 0.0, self.MAX_SPEED))
        
        if self.mission_state == "EXIT_TO_PARK" and self.parking_phase in ("APPROACH", "ALIGN"):
            final_turn = float(np.clip(self.parking_target_turn, TURN_MIN, TURN_MAX))
        else:
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
        
        rear         = self.get_sector_ranges(message, 180, 15)
        side_l = self.get_sector_ranges(message, 90, 20)
        side_r = self.get_sector_ranges(message, -90, 20)

        min_fc = min(front_center) if front_center else 10.0
        min_fl = min(front_left) if front_left else 10.0
        min_fr = min(front_right) if front_right else 10.0
        
        self.min_rear = min(rear) if rear else 10.0
        self.side_left = min(side_l) if side_l else 10.0
        self.side_right = min(side_r) if side_r else 10.0

        self.zone_check_range = min(min_fc, min_fl, min_fr)

        if self.mission_state == "EXIT_TO_PARK" and self.parking_phase > 0:
            self.obstacle_in_front = False
        else:
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

        if self.mission_state == "EXIT_TO_PARK" and self.parking_phase == 0:
            dist = math.sqrt((self.odom_x - self.parking_start_x)**2 + (self.odom_y - self.parking_start_y)**2)
            
            if dist > 2.5:
                if self.side_left < 1.5 and self.side_right > 2.0:
                    self.parking_cone_side = "LEFT"
                    self.parking_phase = 1
                    self.parking_timer_start = time.time()
                    self.get_logger().info("⏱️ LIDAR: Cones detected LEFT. Following cone lane...")
                
                elif self.side_right < 1.5 and self.side_left > 2.0:
                    self.parking_cone_side = "RIGHT"
                    self.parking_phase = 1
                    self.parking_timer_start = time.time()
                    self.get_logger().info("⏱️ LIDAR: Cones detected RIGHT. Following cone lane...")

    def edge_vectors_callback(self, message):
        if self.mission_state in HOLD_STATES:
            return

        if self.mission_state == "EXIT_TO_PARK" and self.parking_phase > 0:
            return

        width = message.image_width
        height = message.image_height

        if width == 0 or height == 0:
            return

        car_x = width / 2.0
        car_y = float(height)

        # 1. 14-SECOND AUTOMATIC TURN TIMEOUT RESTORED
        if self.active_sign_command is not None and getattr(self, 'sign_command_timestamp', 0.0) > 0.0:
            if (time.time() - self.sign_command_timestamp) > 12.0:
                self.active_sign_command = None
                self.in_intersection = False
                self.last_received_sign = None
                self.sign_confirm_count = 0
                self.sign_command_timestamp = 0.0
                self.straight_phase = None
                self.straight_zero_start = 0.0
                self.straight_follow_start = 0.0
                self.get_logger().info("⏱️ Turn completed. Memory automatically reset for next signboard.")

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
                    self.last_received_sign = None
                    self.sign_confirm_count = 0
                    self.sign_command_timestamp = 0.0
                    self.straight_phase = None
                    self.straight_zero_start = 0.0
                    self.straight_follow_start = 0.0
                    self.get_logger().info("🛣️ Lane safely realigned. Memory wiped for next signboard.")

        if is_fork and has_active_sign:
            self.in_intersection = True

        active_count = message.vector_count

        # =========================================================
        # YOUR ORIGINAL STRAIGHT PHASE MACHINE RESTORED
        # =========================================================
        straight_handled = False
        if self.active_sign_command == "STRAIGHT":
            if self.straight_phase == "WAITING_ZERO":
                if message.vector_count == 0:
                    if self.straight_zero_start == 0.0:
                        self.straight_zero_start = time.time()
                    elapsed = time.time() - self.straight_zero_start
                    if elapsed > 0.75:
                        # 1s passed with no edges — switch to FOLLOW_LEFT
                        self.straight_phase = "FOLLOW_LEFT"
                        self.straight_follow_start = time.time()
                        self.get_logger().info("🔀 STRAIGHT: 1s blind done, now following left edge with small offset")
                else:
                    self.straight_zero_start = 0.0
                straight_handled = True

            elif self.straight_phase == "FOLLOW_LEFT":
                elapsed = time.time() - self.straight_follow_start
                if elapsed > 2.5:
                    # Done — full reset
                    self.active_sign_command = None
                    self.straight_phase = None
                    self.in_intersection = False
                    self.last_received_sign = None
                    self.sign_confirm_count = 0
                    self.sign_command_timestamp = 0.0
                    self.straight_zero_start = 0.0
                    self.straight_follow_start = 0.0
                    self.get_logger().info("✅ STRAIGHT: Complete, resuming normal driving")
                else:
                    # Drop v2 (follow left edge only)
                    if v2 is not None:
                        v2 = None
                        active_count = 1
                straight_handled = True

        if not straight_handled and is_fork and has_active_sign:
            if "LEFT" in self.active_sign_command:
                v2 = None
                active_count = 1
            elif "RIGHT" in self.active_sign_command:
                v1 = v2
                v2 = None
                active_count = 1

        # =========================================================
        # YOUR ORIGINAL TARGET CALCULATION RESTORED
        # =========================================================
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

            elif self.active_sign_command == "STRAIGHT" and self.straight_phase == "FOLLOW_LEFT":
                # Smaller offset (0.15) — gentle drift left, not a full turn
                STRAIGHT_OFFSET = width * 0.15
                if avg_x < car_x + 50:
                    target_x, target_y = top_x + STRAIGHT_OFFSET, top_y
                else:
                    target_x, target_y = car_x - 60.0, car_y - 70.0

            elif self.active_sign_command == "RIGHT":
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
            if self.active_sign_command == "STRAIGHT" and self.straight_phase in ("WAITING_ZERO", "FOLLOW_LEFT"):
                # Hold straight ahead during blind crossing
                target_x, target_y = car_x, car_y - 70.0
            elif self.active_sign_command == "LEFT":
                target_x, target_y = car_x - 120.0, car_y - 70.0
            else:
                target_x, target_y = car_x + 120.0, car_y - 70.0

        safe_shift = np.clip(self.obstacle_target_shift, -80.0, 80.0)
        target_x += safe_shift
        target_x = np.clip(target_x, 20.0, width - 20.0)

        dx = car_x - target_x
        dy = car_y - target_y

        dynamic_lookahead = 70.0 + (self.current_actual_speed * 40.0)
        if dy < dynamic_lookahead:
            dy = dynamic_lookahead

        theta_error = math.atan2(dx, dy)
        derivative = theta_error - self.prev_error
        self.prev_error = theta_error

        kp_angle = 0.75; kd_angle = 0.40
        raw_turn = (kp_angle * theta_error) + (kd_angle * derivative)

        # YOUR ORIGINAL SMOOTH TURN AND CLIP RESTORED
        smooth_turn = (0.65 * self.target_turn) + (0.35 * raw_turn)
        self.last_turn_direction = float(np.clip(smooth_turn, -1.0, 1.0))

        approaching_target_qr = False
        if self.sensed_qr:
            if self.mission_state == "SEARCH_PATIENT" and self.sensed_qr == SIGN_TO_PATIENT.get(self.target_letter):
                approaching_target_qr = True
            elif self.mission_state == "SEARCH_HOSPITAL" and self.sensed_qr == self.expected_hospital:
                approaching_target_qr = True

        # --- DYNAMIC SPEED MODULATION ---
        if abs(theta_error) > 0.35:
            self.target_speed = 0.45   
        elif abs(theta_error) > 0.15:
            self.target_speed = 0.80   
        elif approaching_target_qr:
            self.target_speed = 0.50   
        elif self.active_sign_command == "STRAIGHT":
            self.target_speed = 2.00
        elif self.active_sign_command is not None:
            self.target_speed = 1.20   
        else:
            self.target_speed = self.MAX_SPEED

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
        self.last_received_sign = None
        self.sign_confirm_count = 0
        self.sign_command_timestamp = 0.0
        self.straight_phase = None
        self.straight_zero_start = 0.0
        self.straight_follow_start = 0.0

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
                self.set_mission_state("SEARCH_HOSPITAL")
                self.get_logger().info(f"✅ ASSIGNED HOSPITAL: {resolved_hospital} (Route Sign: {self.target_letter})")

        elif self.mission_state == "AT_HOSPITAL_ZONE_WAIT":
            self.patient_index += 1
            resolved_patient = self.resolve_patient_payload(payload)
            self.target_letter = PATIENT_TO_SIGN[resolved_patient] if resolved_patient else self.patient_sequence[min(self.patient_index, len(self.patient_sequence) - 1)]
            self.set_mission_state("SEARCH_PATIENT")
            self.get_logger().info(f"✅ NEXT PATIENT: Proceed to {self.target_letter}")

        elif self.mission_state == "PARKED_WAIT_ACK":
            if payload == "OK": self.set_mission_state("DONE")

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
            if self.sensed_qr == expected:
                if self.zone_stop_timer_start == 0.0:
                    if zone_confirmed:
                        self.zone_stop_timer_start = time.time()
                        self.get_logger().info(f"🎯 ENTERING PATIENT ZONE for {expected}: Rolling forward into center...")

                else:
                    if (time.time() - self.zone_stop_timer_start) >= self.ZONE_STOP_DELAY:
                        self.send_server_update(self.sensed_qr)
                        self.mission_state = "AT_PATIENT_ZONE_WAIT"
                        self.sensed_qr = None
                        self.active_sign_command = None
                        self.in_intersection = False
                        self.last_received_sign = None
                        self.sign_confirm_count = 0
                        self.sign_command_timestamp = 0.0
                        self.straight_phase = None
                        self.straight_zero_start = 0.0
                        self.straight_follow_start = 0.0
                        self.zone_stop_timer_start = 0.0
                        self.get_logger().info(f"🛑 CENTERED IN PATIENT ZONE: Stopped for {expected}")

        elif self.mission_state == "SEARCH_HOSPITAL":
            expected = self.expected_hospital
            if self.sensed_qr == expected:
                if self.zone_stop_timer_start == 0.0:
                    if zone_confirmed:
                        self.zone_stop_timer_start = time.time()
                        self.get_logger().info(f"🎯 ENTERING HOSPITAL ZONE for {expected}: Rolling forward into center...")

                else:
                    if (time.time() - self.zone_stop_timer_start) >= self.ZONE_STOP_DELAY:
                        self.send_server_update(self.sensed_qr)
                        self.sensed_qr = None
                        self.active_sign_command = None
                        self.in_intersection = False
                        self.last_received_sign = None
                        self.sign_confirm_count = 0
                        self.sign_command_timestamp = 0.0
                        self.straight_phase = None
                        self.straight_zero_start = 0.0
                        self.straight_follow_start = 0.0
                        self.zone_stop_timer_start = 0.0
                        self.get_logger().info(f"🛑 CENTERED IN HOSPITAL ZONE: Stopped for {expected}")

                        self.patients_delivered += 1
                        
                        if self.patients_delivered >= 3:
                            self.set_mission_state("EXIT_TO_PARK")
                            self.parking_phase = 0
                            self.parking_start_time = time.time()
                            self.parking_start_x = self.odom_x
                            self.parking_start_y = self.odom_y
                            self.get_logger().info("✅ ALL 3 PATIENTS DELIVERED: Proceeding to Park.")
                        else:
                            self.mission_state = "AT_HOSPITAL_ZONE_WAIT"

    def qr_detection_callback(self, message):
        qr_data = self.normalize_qr_payload(message.data)
        if qr_data in FAKE_HOSPITALS: return
        self.sensed_qr = qr_data
        self.check_zone_arrival()

    def odometry_callback(self, message):
        self.odom_x = message.pose.pose.position.x
        self.odom_y = message.pose.pose.position.y

    def cone_detection_callback(self, message):
        if self.mission_state != "EXIT_TO_PARK" or self.parking_phase != 0:
            return

        dist = math.sqrt((self.odom_x - self.parking_start_x)**2 +
                         (self.odom_y - self.parking_start_y)**2)
        if dist < self.CONE_DETECT_MIN_DIST:
            return

        parts = message.data.split('_')
        if len(parts) >= 2 and parts[0] == "CONES":
            direction = parts[1]  
            if direction in ["LEFT", "RIGHT"]:
                self.parking_cone_side = direction
                self.parking_phase = 1
                self.parking_timer_start = time.time()
                self.get_logger().info(
                    f"📷 CAMERA: Cones detected to {direction}! → Clearing cones to reverse."
                )

    def sign_board_callback(self, message):
        parts = message.data.upper().strip().split('_', 1)
        if len(parts) != 2: 
            return
        
        letter, direction = parts

        if letter != self.target_letter: 
            return

        if getattr(self, 'in_intersection', False) and self.active_sign_command is not None:
            return

        if getattr(self, 'last_received_sign', None) == f"{letter}_{direction}":
            self.sign_confirm_count = getattr(self, 'sign_confirm_count', 0) + 1
        else:
            self.last_received_sign = f"{letter}_{direction}"
            self.sign_confirm_count = 1

        if self.sign_confirm_count >= 2:
            if self.active_sign_command != direction:
                self.active_sign_command = direction
                self.sign_command_timestamp = time.time()
                if direction == "STRAIGHT":
                    self.straight_phase = "WAITING_ZERO"
                    self.straight_zero_start = 0.0
                    self.straight_follow_start = 0.0
                self.get_logger().info(f"🎯 MATCHED TARGET SIGN ({letter})! Intent CONFIRMED & LOCKED: {direction}")


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