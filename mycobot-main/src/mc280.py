"""
@file mc280.py
@brief High-level Python API for controlling a MyCobot 280 arm.
@details
Provides motion primitives (home/sleep poses, gripper control),
PyBullet visualization utilities, and placeholders for kinematics
and trajectory generation.
"""

import os, signal, yaml
import numpy as np
from importlib.util import find_spec
from Lie.group import SE3
from yacs.config import CfgNode
from typing import List
from pymycobot import MyCobot280
from time import sleep, time


## High-level interface to the MyCobot 280 robot arm.
class MC280:
    """
    @class MC280
    @brief High-level controller for the MyCobot 280 arm.
    @details
    Handles hardware initialization, motion commands, basic kinematics calculations, and PyBullet visualization.
    """

    # --- configuration ---
    def __init__(self, mc: MyCobot280, estop: bool = True):
        # Power on the arm
        self.mc = mc
        self.mc.power_on()
        while not self.mc.is_power_on():
            sleep(0.1)
        self.use_estop = estop
        
        # Set IO pins
        self.mc.set_basic_output(2,0)
        sleep(0.1)
        self.mc.set_basic_output(5,0)

        # Joint parameters
        self.encoders_home = [2048, 2048, 2048, 2048, 2048, 2048]
        self.alpha_home = [0, 0, 0, 0, 0, 0]              
        # self.alpha_sleep = [0, 125, -135, -37.26, 9.49, -40.78]
        self.alpha_sleep = [0, 135, -149, -32.0, 0, 45]
        self.alpha_limits = [[-168, -135, -150, -145, -155, -180],
                             [168, 135, 150, 145, 160, 180]]    
        self.alpha_limits_rad = [[np.deg2rad(a) for a in self.alpha_limits[0]],
                                 [np.deg2rad(a) for a in self.alpha_limits[1]]] 
        self.joint_names = ['joint1','joint2','joint3','joint4','joint5','joint6'] 
        self.max_speed = 150 

        # Load transforms
        package_root = os.path.dirname(find_spec("mc280").origin)
        transform_path = os.path.join(package_root, "../utils/transforms.yaml")
        with open(transform_path, "r") as file:
            data = yaml.safe_load(file)
        self.T_BW = data["T_BW"]
        self.T_WB = data["T_WB"]
        self.dhTable = np.asarray(data["dhTable"], dtype=float)

        # PyBullet parameters
        self.model_path = os.path.join(package_root, "../utils/armUrdf/mycobot_280_m5.urdf")

        # Other settings
        if estop:
            signal.signal(signal.SIGINT, self._estop) # sets a callback function on the interrupt signal (Ctrl + C)

    def set_joint_limits(self, new_alpha, new_usec=None, new_orient=None):
        """
        @brief Update the joint limit array.
        @param new_alpha  New joint limit matrix [[lo1..lo6],[hi1..hi6]] in degrees.
        """
        self.alpha_limits = new_alpha

    # --- motion ---
    def goto_sleep(self, timeout=10):
        """
        @brief Move arm to the sleep configuration.
        @param timeout Maximum wait time before target is reached
        """
        self.mc.clear_error_information()
        self._send_angles(self.alpha_sleep, 10, timeout=timeout)

    def goto_home(self, timeout=10):
        """
        @brief Move arm to the home configuration.
        """
        self.mc.clear_error_information()
        self.set_encoders(self.encoders_home, 5)
        sleep(4)

    def set_joints_by_time(self, alpha, time=None, timeout=10):
        """
        @brief Move to a target joint configuration in a specified time. 
        @param alpha  Target joint angles (deg).
        @param time   Desired motion time (s). Must be > 0.
                Used to compute speed as:
                speed = (max_delta / time) / max_speed * 100,
                where max_delta is the largest joint change (deg).
                Clamped to [1,100].
        @param timeout Maximum wait time before target is reached
        @throws ValueError if time <= 0.
        """
        if time <= 0:
            raise ValueError("`time` must be positive.")
        
        # Calculate
        current_alpha = self.measure_joints()
        deltas = [abs(a - b) for a, b in zip(alpha, current_alpha)]
        max_delta = max(deltas)
        if max_delta == 0:
            return
        speed = round((max_delta / time) / self.max_speed * 100)
        speed = max(1, min(100, speed))
        self._send_angles(alpha, speed, timeout=timeout)

    def set_joints(self, alpha, speed=20, timeout=10):
        """
        @brief Move to a target joint configuration at a given speed.
        @param alpha A list of six target joint angles (deg).
        @param speed Percentage of maximum joint speed of 150 degrees per second [1–100].
        @param timeout Maximum wait time before target is reached
        """
        self._send_angles(alpha, speed, timeout=timeout)


    def set_flange(self, pose, speed=20, timeout=10):
        """
        @brief Move to a target joint configuration at a given speed. 
        @param pose Flange pose in the base frame as a list [x_mm, y_mm, z_mm, rx_deg, ry_deg, rz_deg]
        @param speed Percentage of maximum joint speed of 150 degrees per second [1–100].
        @param timeout Maximum wait time before target is reached

        The base frame is defined as:
        - Origin at the XY center of the base. The Z=0 reference is the bottom edge of the robot base.
        - Display screen on the robot base faces the negative X direction.
        - Display screen's left is the positive Y direction.
        - Upward from the base is the positive Z direction.

        The flange frame is defined as:
        - Origin at the center of the flange’s surface (where the gripper attaches).
        - The type-C port on link 6 faces the negative X direction.
        - The flange faces the positive Z direction.
        - z_mm measures height from the bottom edge of the robot base (excluding the vacuum mounts).
        """
        self._send_coords(pose, speed=speed, timeout=timeout)

    def set_color(self, r, g, b):
        """
        @brief Set the color of the display on the back of joint 6

        @param 
            - r = red
            - b = blue
            - g = green
        @return 1 on completion
        """
        return self.mc.set_color(r, g, b)

    def shutdown(self):
        """
        @brief Move to sleep pose, close gripper, release servos and power off.
        """
        self.goto_sleep()
        self.release_motors()
        sleep(0.5)
        self.mc.power_off()

    def release_motors(self):
        """
        @brief Release all servo motors (no holding torque).
        @return 1 on completion
        """
        return self.mc.release_all_servos()
    
    def lock_motors(self):
        """
        @brief Lock all servo motors.
        @return 1 on completion
        """
        return self.mc.focus_all_servos()

    def increment_joint(self, joint_id: int, increment: float, speed: int = 20):
        """
        @brief Incrementally move a single joint by angle.
        @param joint_id Joint number (1–6).
        @param increment Angle increment relative to current joint position.
        @param speed Movement speed (1–100).
        @return 1 on completion
        """
        self.mc.clear_error_information()
        return self.mc.jog_increment_angle(joint_id, increment, speed)


    def increment_coord(self, axis_id: int, increment: float, speed: int = 20):
        """
        @brief Incrementally move along a single Cartesian axis.
        @param axis_id Axis number (1–6) corresponding to X, Y, Z, Rx, Ry, Rz.
        @param increment Incremental movement relative to current coordinate position.
        @param speed Movement speed (1–100).
        @return 1 on completion
        """
        self.mc.clear_error_information()
        return self.mc.jog_increment_coord(axis_id, increment, speed)
    
    def increment_coords(self, increments: list, speed: int = 20, timeout=10):
        """
        @brief Incremental all six coordinates
        @param increments A list of six integers [dx_mm, dy_mm, dz_mm, drx_deg, dry_deg, drz_deg]
        @param speed Movement speed (1–100).
        @param timeout Maximum wait time until target is reached. 
        @return 1 on completion
        """
        current_coords = self.measure_flange()
        new_coords = [a + b for (a,b) in zip(current_coords, increments)]
        return self._send_coords(new_coords, speed=speed, timeout=timeout)
    
    def set_encoder(self, joint_id, encoder_val, speed = 20):
        """
        @brief Set encoder positions
        @param encoder_val An int in [0, 4096]
        @param speed A percentage of the maximum joint speed of 150 degrees per second. An int in [1-100]
        @param joint_id 1-6 if setting a single joint
        @return 1 on completion
        """
        return self.mc.set_encoder(joint_id, encoder_val, speed)
    
    def set_encoders(self, encoder_vals, speed = 20):
        """
        @brief Set all six encoder positions
        @param encoder_val An list of six ints in [0, 4096]
        @param speed A percentage of the maximum joint speed of 150 degrees per second. An int in [1-100]
        @return 1 on completion
        """
        return self.mc.set_encoders(encoder_vals, speed)


    # --- measure joints & pose --- 
    def measure_joints(self, rad=False):
        """
        @brief Read current joint angles.
        @param rad If true, return in radians.
        @return List of joint angles.

        The increment_joint() call is to resolve a weird bug, where right after setting gripper
        joint and flange measurements can't be made until another set joint or flange command is called
        """
        self.increment_joint(1, 0) 
        alpha = self.mc.get_angles()

        # Sometimes mc.get_angles() returns -1, so keep invoking it until a result is obtained
        count = 1
        while alpha is None or alpha == -1:
            print("Failed to measure joints. Retrying...")
            alpha = self.mc.get_angles()

            count += 1
            if count > 5:
                raise Exception("Unable to measure joint angles.")
        return np.deg2rad(alpha) if rad else alpha

    def measure_flange(self):
        """
        @brief Query MyCobot for the flange pose as SE3.Homog.
        @param mc  MyCobot280 instance.
        @return Flange pose in the base frame as a list [x_mm, y_mm, z_mm, rx_deg, ry_deg, rz_deg].

        The base frame is defined as:
        - Origin at the XY center of the base. The Z=0 reference is the bottom edge of the robot base.
        - Display screen on the robot base faces the negative X direction.
        - Display screen's left is the positive Y direction.
        - Upward from the base is the positive Z direction.

        The flange frame is defined as:
        - Origin at the center of the flange’s surface (where the gripper attaches).
        - The type-C port on link 6 faces the negative X direction.
        - The flange faces the positive Z direction.
        - z_mm measures height from the bottom edge of the robot base (excluding the vacuum mounts).
        
        The increment_joint() call is to resolve a weird bug, where right after setting gripper
        joint and flange measurements can't be made until another set joint or flange command is called
        """
        self.increment_joint(1, 0) 
        coords = self.mc.get_coords()

        # Sometimes mc.get_coords() returns -1, so keep invoking it until a result is obtained
        count = 1
        while coords is None or coords == -1:
            print("Failed to measure flange. Retrying...")
            coords = self.mc.get_coords()

            count += 1
            if count > 5:
                raise Exception("Unable to measure end effector pose.")
        return coords
    
    def measure_encoders(self, joint_id: int|None = None):
        """
        @brief Measure encoder positions
        @param joint_id 1-6 if getting a single joint; None to get all.
        @return If joint_id is None: an int in [0, 4096]. If joint_id is not None: a list of 6 ints in [0, 4096]
        """
        if joint_id is None:
            return self.mc.get_encoders()
        return self.mc.get_encoder(joint_id)
    
    
    def measure_servo_speeds(self):
        """
        @brief Measure servo speeds 
        @return A list of six speed values in steps/sec
        """
        return self.mc.get_servo_speeds()
    
    # --- visualization ---
    def display_arm(self, alpha=None):
        """
        @brief Display the robot in a PyBullet GUI at the specified joint configuration.
        @param alpha Optional array of joint angles in radians. If None, the current
                    measured angles are used. Blocks and continuously steps the simulation.
        """
        import pybullet as p
        import pybullet_data

        if alpha is None:
            alpha = self.measure_joints(rad=True)
        # set up pybullet
        p.connect(p.GUI)
        p.setAdditionalSearchPath(pybullet_data.getDataPath())
        p.resetDebugVisualizerCamera(cameraDistance=0.5,
                                    cameraYaw=45,
                                    cameraPitch=-30,
                                    cameraTargetPosition=[0,0,0.2]) # zoom in camera
        p.loadURDF("plane.urdf")
        p.setGravity(0, 0, -9.81)
        model = p.loadURDF(self.model_path, useFixedBase=True, flags=p.URDF_USE_SELF_COLLISION)

        # initialize joint states
        for i, angle in enumerate(alpha):
            p.resetJointState(model, i+1, angle)

        while p.isConnected():
            p.stepSimulation()
            sleep(1/240.0)

    def display_arm_with_sliders(self, alpha=None):
        """
        @brief Launch a PyBullet GUI with interactive sliders to control each joint.
        @param alpha Optional array of initial joint angles in radians. If None, the
                    current measured angles are used. Updates the joint states live
                    as sliders are moved.
        """
        import pybullet as p
        import pybullet_data

        if alpha is None:
            alpha = self.measure_joints(rad=True)

        # set up pybullet
        p.connect(p.GUI)
        p.setAdditionalSearchPath(pybullet_data.getDataPath())
        p.resetDebugVisualizerCamera(cameraDistance=0.5,
                                    cameraYaw=45,
                                    cameraPitch=-30,
                                    cameraTargetPosition=[0,0,0.2])
        p.loadURDF("plane.urdf")
        p.setGravity(0, 0, -9.81)

        model = p.loadURDF(self.model_path, useFixedBase=True, flags=p.URDF_USE_SELF_COLLISION)
        
        
        # set up slider
        slider_ids = []
        for i, joint in enumerate(self.joint_names):
            lower = self.alpha_limits_rad[0][i]
            upper = self.alpha_limits_rad[1][i]
            slider_ids.append(
                p.addUserDebugParameter(paramName=joint,
                                        rangeMin=lower,
                                        rangeMax=upper,
                                        startValue=alpha[i])
            )
        
        # initialize joint states
        for i, angle in enumerate(alpha):
            p.resetJointState(model, i+1, angle)
        
        # keep reading from sliders and updating the simulation
        while p.isConnected():
            for i in range(6):
                angle = p.readUserDebugParameter(slider_ids[i])
                p.resetJointState(model, i+1, angle)
            p.stepSimulation()
            sleep(1/240.0)

    # --- kinematics placeholders ---
    def fk(self, angles: list):
        """
        @brief Compute forward kinematics of the flange pose.
        @param alpha_rad List of joint angles in degrees.
        @return SE3.Homog pose of the end-effector with the angles in radians.

        Note the z coordinate calculated from DH table is off by about -8.15 mm compared to 
        the built-in fk solver used in measure_flange(), hence the +8.15 line at the end.
        """
        dh = self.dhTable
        angles = np.asarray(angles, float)

        d, a, alpha, offset = dh[:,0], dh[:,1], dh[:,2], dh[:,3]

        theta_rad = np.deg2rad(angles + offset)
        alpha_rad = np.deg2rad(alpha)

        T = np.eye(4)
        for di, ai, alphai, thetai in zip(d, a, alpha_rad, theta_rad):
            ct, st = np.cos(thetai), np.sin(thetai)
            ca, sa = np.cos(alphai), np.sin(alphai)

            A = np.array([[ ct, -st*ca,  st*sa, ai*ct],
                      [ st,  ct*ca, -ct*sa, ai*st],
                      [  0,     sa,     ca,    di],
                      [  0,      0,      0,     1]])
            T = T @ A
        
        R, x = T[0:3, 0:3], T[0:3, 3].reshape(3,1)
        x[2] += 8.15
        return SE3.Homog(R=R, x=x)

    def ik(self, pose: list|SE3.Homog):
        """
        @brief Compute inverse kinematics using the native kinematics solver.
        @param pose Desired flange pose as an SE3.Homog() or list [x_mm, y_mm, z_mm, rx_deg, ry_deg, rz_deg] in the base frame.
        @return list of joint angles in degrees

        Note mc.solve_inv_kinematics() will sometimes fail and return [82.41,82.41,82.41,82.41,82.41,82.41], 
        which is obviously invalid. This is checked for by the while loop.
        """
        if isinstance(pose, SE3.Homog):
            pose = self.SE3_to_pose(pose)
        current_angles = self.measure_joints()
        joints = self.mc.solve_inv_kinematics(pose, current_angles)

        max_tries = 3
        while not isinstance(joints, list) or all(joint > 80 and joint < 85 for joint in joints):
            joints = self.mc.solve_inv_kinematics(pose, current_angles=current_angles)
            max_tries -= 1
            if max_tries == 0:
                break
        return joints
    
    # --- reference frames ---
    def set_reference(self, coords, ref_type: str):
        """
        @brief Set tool or world reference coordinates.
        @param coords [x, y, z, rx, ry, rz] coordinate values.
        @param ref_type "tool" or "world".

        The coordinates will be permanently updated. The default values are [0,0,0,0,0,0]. Note that from preliminary testing changing the coordinate has no impact on the robot.
        """
        ref_type = ref_type.lower()
        if ref_type == "tool":
            self.mc.set_tool_reference(coords)
        elif ref_type == "world":
            self.mc.set_world_reference(coords)
        else:
            print("Error: ref_type must be 'tool' or 'world'")


    def get_reference(self, ref_type: str):
        """
        @brief Get tool or world reference coordinates.
        @param ref_type "tool" or "world".
        """
        ref_type = ref_type.lower()
        if ref_type == "tool":
            self.mc.get_tool_reference()
        elif ref_type == "world":
            self.mc.get_world_reference()
        else:
            print("Error: ref_type must be 'tool' or 'world'")

    def pose_to_SE3(self, pose: list):
        """
        @brief Convert a list of coordinates to SE3.Homog.
        @param pose [x_mm, y_mm, z_mm, rx_deg, ry_deg, rz_deg].
        @return SE3.Homog pose in the base frame in radians.
        """
        x_mm, y_mm, z_mm, rx_deg, ry_deg, rz_deg = pose
        xyz_m = np.array([[x_mm], [y_mm], [z_mm]])
        
        # Euler → rotation matrix
        Rz, Ry, Rx = np.deg2rad([rz_deg, ry_deg, rx_deg])
        cz, sz = np.cos(Rz), np.sin(Rz)
        cy, sy = np.cos(Ry), np.sin(Ry)
        cx, sx = np.cos(Rx), np.sin(Rx)
        RzM = np.array([[cz, -sz, 0],
                        [sz,  cz, 0],
                        [ 0,   0, 1]])
        RyM = np.array([[cy,  0, sy],
                        [ 0,  1,  0],
                        [-sy, 0, cy]])
        RxM = np.array([[1,  0,   0],
                        [0, cx, -sx],
                        [0, sx,  cx]])

        R = RzM @ RyM @ RxM

        return SE3.Homog(R=R, x=xyz_m)
    
    def SE3_to_pose(self, se3: SE3.Homog):
        """
        @brief Convert a list of coordinates to SE3.Homog.
        @param se3 SE3.Homog pose in the base frame and radians.
        @return [x_mm, y_mm, z_mm, rx_deg, ry_deg, rz_deg].
        """
        x_mm, y_mm, z_mm = se3.getTranslation()
        rx_deg, ry_deg, rz_deg = np.rad2deg(se3.getRPY())
        return [x_mm, y_mm, z_mm, rx_deg, ry_deg, rz_deg]

    # --- error ---
    def get_error(self):
        """
        @brief Query MyCobot for error information and return a human-readable message. Note there is an error when the display on joint 5 turns blue.
        @return str  Error description.
                    - "No error." if return code = 0.
                    - Joint limit error if 1–6.
                    - Collision protection if 16–19.
                    - "IK has no solution." if 32.
                    - "Linear motion has no adjacent solution." if 33–34.
                    - "Unknown error code <n>" otherwise.
        """
        code = self.mc.get_error_information()

        if code == 0:
            return "No error."
        elif 1 <= code <= 6:
            return f"Joint {code} exceeds limit position."
        elif 16 <= code <= 19:
            return "Collision protection triggered."
        elif code == 32:
            return "IK has no solution."
        elif code in (33, 34):
            return "Linear motion has no adjacent solution."
        else:
            return f"Unknown error code {code}"
    
    def clear_error(self):
        """
        @brief Clear error message. The display on joint 5 should turn green to signal no error.
        """
        self.mc.clear_error_information()

    # --- helpers ---
    def _send_angles(self, angles, speed, timeout):
        self.mc.clear_error_information()
        if self.use_estop:
            status = self.mc.send_angles(angles, speed)
            self._block_till_reached(angles, flag=0)
        else:
            status = self.mc.sync_send_angles(angles, speed, timeout)
        if status != 1:
            error = self.get_error()
            print(f"[WARNING] {error}")
        return status
    
    def _send_coords(self, coords, speed, timeout):
        self.mc.clear_error_information()
        if self.use_estop:
            status = self.mc.send_coords(coords, speed)
            self._block_till_reached(coords, flag=1)
        else:
            status = self.mc.sync_send_coords(coords, speed, timeout)
        if status != 1:
            error = self.get_error()
            print(f"[WARNING] {error}")
        return status
    
    def _block_till_reached(self, goal, flag=0):
        """
        @brief Block until the arm reaches a goal configuration.
        @param goal Joint angles (deg).
        @param flag 0 = joint angles, 1 = flange position
        """
        if isinstance(goal, np.ndarray):
            goal = goal.tolist()

        timeout = 15
        start = time()
        while True:
            reached = self.mc.is_in_position(goal, flag)
            if reached == 1 or time() - start > timeout:
                break
            sleep(0.1)
        sleep(0.5)


    def _estop(self, signum, frame):
        """
        @brief Emergency stop all joint movements. Triggered by the interrupt signal (Ctrl + C)
        """
        print("\n Robot Stopped!")
        try:
            self.mc.stop()
        except Exception as e:
            print("[MC280] Arm stop error:", e)
            
        exit(1) # remove if robot class exits

class Adaptive:
    def __init__(self, mc: MyCobot280, cfg: CfgNode):
        self.mc = mc
        self.mc.init_gripper()
        self.cfg = cfg
        self.speed = self.cfg.others.speed
    
    def set(self, pct_open):
        """
        @brief Move the gripper to a specified open percentage.

        Sends a command to the gripper to reach the desired opening
        percentage and blocks until the gripper finishes moving,
        printing "moving" while it is in motion.

        @param pct_open Desired opening of the gripper in percent:
            - 0   = fully closed
            - 100 = fully open

        The speed and force parameters passed to
        mc.set_gripper_value() are fixed at 20 and 1 respectively.
        """
        if pct_open < self.cfg.command_range[0]:
            pct_open = self.cfg.command_range[0]
        elif pct_open > self.cfg.command_range[1]:
            pct_open = self.cfg.command_range[1]
        self.mc.set_gripper_value(pct_open, self.speed, 1)
    
    def measure(self):
        """
        @brief Measure the gripper's current state.
        @return An integer indicating the percent open.
        """
        gripper_value = self.mc.get_gripper_value(1)
        while gripper_value == -1:
            gripper_value = self.mc.get_gripper_value(1)
        return gripper_value
    
    def calibrate(self):
        """
        @brief Set the current gripper position to be 100 (i.e. 100 percent open)
        @return 1 if success.
        """
        return self.mc.set_gripper_calibration()

class Flexible:
    def __init__(self, mc: MyCobot280, cfg: CfgNode):
        self.mc = mc
        self.mc.init_gripper()
        self.cfg = cfg
        self.speed = self.cfg.others.speed
    
    def set(self, pct_open):
        """
        @brief Move the gripper to a specified open percentage.

        Sends a command to the gripper to reach the desired opening
        percentage and blocks until the gripper finishes moving,
        printing "moving" while it is in motion.

        @param pct_open Desired opening of the gripper in percent:
            - 0   = fully closed
            - 100 = fully open

        The speed and force parameters passed to
        mc.set_gripper_value() are fixed at 20 and 1 respectively.
        """
        if pct_open < self.cfg.command_range[0]:
            pct_open = self.cfg.command_range[0]
        elif pct_open > self.cfg.command_range[1]:
            pct_open = self.cfg.command_range[1]
        self.mc.set_gripper_value(pct_open, self.speed, 4)
    
    def measure(self):
        """
        @brief Measure the gripper's current state.
        @return An integer indicating the percent open.
        """
        gripper_value = self.mc.get_gripper_value(4)
        while gripper_value == -1:
            gripper_value = self.mc.get_gripper_value(4)
        return gripper_value
    
    def calibrate(self):
        """
        @brief Set the current gripper position to be 100 (i.e. 100 percent open)
        @return 1 if success.
        """
        return self.mc.set_gripper_calibration()
    
class Suction:
    def __init__(self, mc: MyCobot280, cfg: CfgNode):
        self.mc = mc
        self.cfg = cfg
        self.pump_pin = self.cfg.others.pump_pin
        self.valve_pin = self.cfg.others.valve_pin
        self.is_on = False
    
    def set(self, on: bool):
        """
        @brief Turn suction ON or OFF.

        on = True  → suction ON  (pump on, valve on)
        on = False → suction OFF (pump off, valve off)
        When the valve is ON, its plastic end opens and metal end closes.
        """
        if on:
            self.mc.set_basic_output(self.valve_pin, 1)
            sleep(0.1)
            self.mc.set_basic_output(self.pump_pin, 1)
            self.is_on = True
        else:
            self.mc.set_basic_output(self.pump_pin, 0)
            sleep(0.1)
            self.mc.set_basic_output(self.valve_pin, 0)
            self.is_on = False
    
    def measure(self):
        """
        @brief Measure the gripper's current state.
        @return True if both pump and valve are ON.

        This function doesn't directly measure the GPIO pins 
        because doing so would turn off the pump and the valve.
        """
        return self.is_on
        
    