import mc280
import numpy as np
import os, sys, json, time, threading, yaml, readchar
from pymycobot import MyCobot280
from time import sleep
from Lie.group import SE3
from typing import List
from pathlib import Path
from datetime import datetime
from yacs.config import CfgNode
from ivapy.Configuration import AlgConfig
from importlib.util import find_spec

class Manipulator:
    def __init__(self, mc: MyCobot280):
        whole_cfg = self.load_cfg()
        self.hand = self.build_gripper(mc, whole_cfg)
        self.hand_cfg = whole_cfg[whole_cfg.selected]
        
        self.arm = mc280.MC280(mc)
        self.recorder = Recorder(mc)

        self.T_FG = np.array(self.hand_cfg.T_FG)
        self.tcp_G = np.array(self.hand_cfg.default_tcp + [1])

    def fk(self, angles):
        """
        @brief Compute forward kinematics of tcp
        @param alpha_rad List of joint angles in degrees.
        @return SE3.Homog pose of the end-effector with the angles in radians.

        Note TCP is defined in the gripper frame and shares orientation with gripper.
        """
        T_BF_se3 = self.arm.fk(angles)
        T_BF = self.se3_to_matrix(T_BF_se3)
        T_BG = T_BF @ self.T_FG
        
        tcp_B = T_BG @ self.tcp_G
        tcp_B = np.array(tcp_B[0:3]).reshape(3,1)

        R_BG = T_BF_se3.getRotation() @ self.T_FG[0:3,0:3]

        return SE3.Homog(R=R_BG, x=tcp_B)

    def ik(self, pose: list|SE3.Homog):
        """
        @brief Compute inverse kinematics of the tcp pose in base frame.
        @param pose Desired tcp pose in base frame as a list [x_mm, y_mm, z_mm, rx_deg, ry_deg, rz_deg] in the base frame.
        @return list of joint angles in degrees

        Note TCP is defined in the gripper frame and shares orientation with gripper.
        """
        if isinstance(pose, list):
            pose = self.arm.pose_to_SE3(pose)
        T_BT = self.se3_to_matrix(pose) # base to TCP 

        T_GT = np.eye(4) # gripper frame to TCP 
        T_GT[0:3, 3] = self.tcp_G[0:3]

        T_FT = self.T_FG @ T_GT
        T_TF = np.linalg.inv(T_FT)
        T_BF = T_BT @ T_TF

        T_BF_se3 = SE3.Homog(M=T_BF)
        flange_pose = self.arm.SE3_to_pose(T_BF_se3)
        joints = self.arm.ik(flange_pose)
        if not isinstance(joints, list) or all(joint > 80 for joint in joints):
            raise Exception("Failed to find IK solution")
        return joints

    def release(self, top_xyz, release_height=50):
        """
        @brief Release an object at a target point.

        Moves the robot to a hover pose above the target, descends to a
        release position, opens the gripper, and returns to the hover height.

        @param top_xyz [x_mm, y_mm, z_mm] in world coordinates representing
            the top of the desired release point.
        @param release_height Vertical offset (in mm) above the release point
            used for safe approach and retreat. Default is 50 mm.
        """
        # Convert from world to base frame
        top_xyz_W = np.array(top_xyz + [1.0], dtype=float)
        top_xyz_B_h = self.arm.T_BW @ top_xyz_W
        top_xyz_B = top_xyz_B_h[:3].tolist()

        orientation = [-90, 0, -90]  # release from the top (expressed in gripper frame)

        finger_length = self.hand_cfg.default_finger_length

        # Compute TCP positions for release and hover
        tcp_xyz_release = self.tcp_from_top(
            top_xyz_B,
            dz=release_height,
        )

        tcp_release = tcp_xyz_release + orientation
        joints_release = self.ik(tcp_release)


        # Descend to release position
        self.arm.set_joints(joints_release, speed=5)
        sleep(1)

        # Open gripper to drop the object
        self.hand.set(self.hand.cfg.command_range[1])
        sleep(1)

    def capture(self, top_xyz, hover_height=30):
        """
        @brief Perform a pick motion above a target point and grasp the object.

        Moves the robot to a hover pose above the target, then descends to a
        grasp position, closes the gripper, and returns to the hover height.

        @param top_xyz Target point in world coordinates [x_mm, y_mm, z_mm]
            representing the top of the object.
        @param hover_height Vertical offset (in mm) above the grasp point used
            for safe approach and retreat. Default is 50 mm.

        The TCP used for positioning is defined at the grasp center (midpoint
        between the two gripper fingers) and shares orientation with the gripper.
        """
        # Convert from world to base frame
        top_xyz_W = np.array(top_xyz + [1.0], dtype=float)
        top_xyz_B_h = self.arm.T_BW @ top_xyz_W
        top_xyz_B = top_xyz_B_h[:3].tolist()
        orientation = [-90, 0, -90]  # capture from the top (expressed in gripper frame)

        finger_length = self.hand_cfg.default_finger_length

        # Compute TCP positions for capture and hover
        tcp_xyz_capture = self.tcp_from_top(
            top_xyz_B,
            dz=-(finger_length - 15),   # descend toward object
        )
        tcp_xyz_hover = self.tcp_from_top(
            top_xyz_B,
            dz=hover_height,            # raise for safe approach/retreat
        )

        tcp_capture = tcp_xyz_capture + orientation
        joints_capture = self.ik(tcp_capture)

        tcp_hover = tcp_xyz_hover + orientation
        joints_hover = self.ik(tcp_hover)

        # Move above the grasp point
        self.arm.set_joints(joints_hover, speed=5)
        sleep(2)
        # print(f"Moved to hover position. TCP = {tcp_hover}. Joints = {joints_hover}")
        # Open gripper before descending
        self.hand.set(self.hand.cfg.command_range[1])
        sleep(1)

        # Descend to grasp position
        self.arm.set_joints(joints_capture, speed=5)
        sleep(2)
        # print(f"Moved to capture position. TCP = {tcp_capture}. Joints = {joints_capture}")
        # Close gripper to grasp the object
        self.hand.set(self.hand.cfg.command_range[0])
        sleep(2)

        # Return to hover height
        self.arm.set_joints(joints_hover, speed=5)


    def record_cli(self):
        self.recorder.start_cli()

    def jog_cli(self, step_lin=20.0, step_rot=20.0, step_gripper=20, speed=50):
        """
        @brief Minimal cross-platform interactive jog control (using readchar).
        @param step_lin Step size for linear motion in mm.
        @param step_rot Step size for angular motion in deg.
        @param speed Speed as a percentage of the maximum joint speed.

        Keys:
        x / X : ±X
        y / Y : ±Y
        z / Z : ±Z
        r / R : ±Rx
        p / P : ±Ry
        w / W : ±Rz
        g / G : open/close gripper
        q     : quit
        """
        print("\r=== Interactive Jog Mode ===")
        print("x/X: ±X, y/Y: ±Y, z/Z: ±Z, r/R: ±Rx, p/P: ±Ry, w/W: q: quit\n")
        print(f"Step sizes: {step_lin:.1f} mm, {step_rot:.1f}°, {step_gripper}%, speed={speed}%\n")

        def show_pose():
            pose = self.arm.measure_flange()
            msg = (f"\rCurrent flange pose [x y z rx ry rz]: "
                f"[{pose[0]:7.2f}, {pose[1]:7.2f}, {pose[2]:7.2f}, "
                f"{pose[3]:7.2f}, {pose[4]:7.2f}, {pose[5]:7.2f}]")
            sys.stdout.write(msg)
            sys.stdout.flush()

        show_pose()

        while True:
            ch = readchar.readkey()
            if ch == "q":
                print("\nExiting jog mode.")
                break
            elif ch == "x": self.arm.increment_coord(1, +step_lin, speed)
            elif ch == "X": self.arm.increment_coord(1, -step_lin, speed)
            elif ch == "y": self.arm.increment_coord(2, +step_lin, speed)
            elif ch == "Y": self.arm.increment_coord(2, -step_lin, speed)
            elif ch == "z": self.arm.increment_coord(3, +step_lin, speed)
            elif ch == "Z": self.arm.increment_coord(3, -step_lin, speed)
            elif ch == "r": self.arm.increment_coord(4, +step_rot, speed)
            elif ch == "R": self.arm.increment_coord(4, -step_rot, speed)
            elif ch == "p": self.arm.increment_coord(5, +step_rot, speed)
            elif ch == "P": self.arm.increment_coord(5, -step_rot, speed)
            elif ch == "w": self.arm.increment_coord(6, +step_rot, speed)
            elif ch == "W": self.arm.increment_coord(6, -step_rot, speed)
            elif ch == "g": self.hand.set(self.hand.measure() + step_gripper)
            elif ch == "G": self.hand.set(self.hand.measure() - step_gripper)
            else:
                sys.stdout.write("\n")  # end the live line cleanly
                sys.stdout.flush()
                print(f"Ignored key: {repr(ch)}")

            sleep(0.05)
            show_pose()


    def load_cfg(self):
        pkg_root = os.path.dirname(find_spec("mc280").origin)
        gripper_path = os.path.join(pkg_root, "../utils/gripperConfigs.yaml")
        with open(gripper_path, "r") as file:
            config_dict = yaml.safe_load(file)
        config = AlgConfig(config_dict)
        return config

    def build_gripper(self, mc: MyCobot280, cfg: CfgNode):
        if cfg.selected == "adaptive":
            return mc280.Adaptive(mc, cfg.adaptive)
        elif cfg.selected == "flexible":
            return mc280.Flexible(mc, cfg.flexible)
        elif cfg.selected == "suction":
            return mc280.Suction(mc, cfg.suction)
        raise Exception("Unrecognized gripper type")
    
    # Helpers
    def tcp_from_top(self, top_xyz_B, dz):
        """
        @brief Given the top-of-object point in base frame, compute a single TCP xyz
        offset in z by `dz` (mm), clamped to min grip height.

        @param top_xyz_B: [x_mm, y_mm, z_mm] in base frame.
        @param dz: vertical offset in mm relative to top point
            (positive = above, negative = below).
        """
        x, y, z_top = top_xyz_B
        z_tcp = z_top + dz
        z_tcp = max(z_tcp, self.hand.cfg.min_grip_height)
        return [x, y, z_tcp]
    
    def se3_to_matrix(self, se3: SE3.Homog):
        """
        @brief Convert a list of coordinates to SE3.Homog.
        @return [x_mm, y_mm, z_mm, rx_deg, ry_deg, rz_deg].
        """
        matrix = np.eye(4)
        matrix[0:3,0:3] = se3.getRotation()
        matrix[0:3,3] = se3.getTranslation()[0:3,0]
        return matrix
    

class Recorder():
    """@brief Record and replay MyCobot 280 joint trajectories.
    @details Records raw servo positions and speeds, supports playback (single/loop),
             and interactive save/load via simple keyboard menu.
    """

    def __init__(self, mc: MyCobot280) -> None:
        """@brief Construct a Recorder and connect to MyCobot.
        @note Uses /dev/ttyACM0 @ 115200 by default.
        """
        super().__init__()
        self.mc = mc
        self._recording = False
        self._playing = False
        self._record_list: List = []  # each: [pos (ticks[]), speeds (int[]), dt (s)]
        self._record_t: threading.Thread | None = None
        self._play_t: threading.Thread | None = None

    # ----------------- record / play -----------------
    def start_record(self):
        """@brief Begin recording servo positions/speeds.
        @details Clears previous samples, releases motors (drag/teach),
                 and spawns a background thread to poll the robot.
        @note Samples are encoder ticks (not degrees).
        """
        self._record_list = []
        self._recording = True
        self.mc.release_all_servos()

        def _worker():
            while self._recording:
                t0 = time.time()
                pos = self.mc.get_encoders()
                speeds = self.mc.get_servo_speeds()
                dt = time.time() - t0
                if isinstance(pos, list):
                    self._record_list.append([pos, speeds, dt])
                    self._echo(f"Recording… samples={len(self._record_list)}")
                time.sleep(0.1)
        self._echo("Recording started.")
        self._record_t = threading.Thread(target=_worker, daemon=True)
        self._record_t.start()

    def stop_record(self):
        """@brief Stop recording and re-enable holding torque.
        @note Joins the record thread for a clean shutdown.
        """
        self.mc.focus_all_servos()
        if self._recording:
            self._recording = False
            self._record_t.join()
            self._echo("Recording stopped.")

    def play(self, stride: int = 1):
        """@brief Replay the recorded trajectory.
        @param stride Keep every Nth sample (down-sampling). Default: 3.
        @note Sends raw encoder ticks with per-servo speeds via set_encoders_drag.
        """
        if not self._record_list:
            print("\n[!] No recorded data. Record first with 'r'.\n")
            return

        max_stride = max(1, len(self._record_list))
        if stride < 1 or stride > max_stride:
            print("Invalid stride length; using default.")
            stride = 1

        samples = self._record_list[::stride]
        if not samples:
            print("\n[!] No samples after downsampling.\n")
            return

        self._echo("Moving to starting position.")
        start_pos, _, _ = samples[0]
        status = self.mc.set_encoders(start_pos, 10)
        time.sleep(4)


        # Check if start position is reached
        curr_pos = self.mc.get_encoders()
        pos_diff = sum([abs(curr_pos[i]-start_pos[i]) for i in range(6)])
        if pos_diff > 80:
            self._echo("\n[ERR] Unable to reach start position. Certain encoder values might be too close to limit.")
            return
        self._echo("Playback started.")
        for i in range(1, len(samples)):
            pos, speeds, _ = samples[i]
            print(pos, speeds)
            self.mc.set_encoders(pos, speeds)
            time.sleep(0.2)
        self._echo("Playback finished.\n")
        time.sleep(1)

    def loop_play(self):
        """@brief Continuously replay until stopped."""
        if not self._record_list:
            print("\n[!] No recorded data. Record first with 'r'.\n")
            return

        self._playing = True

        def _loop():
            while self._playing:
                self.play()

        print("\n[LOOP] Starting looped playback. Press 'P' again to stop.\n")
        self._play_t = threading.Thread(target=_loop, daemon=True)
        self._play_t.start()

    def stop_loop_play(self):
        """@brief Stop looped playback."""
        if self._playing:
            self._playing = False
            self._play_t.join()
            print("\n[STOP] Looped playback stopped.\n")

    # ----------------- save and load trajectories -----------------
    def save(self, save_path: str | None = None):
        """@brief Save recorded samples to a JSON file.
        @param save_path Optional path; if omitted, prompts for dir/name.
        @note Creates parent directories; prints resolved path on success.
        """
        if not self._record_list:
            print("\n[!] No data to save.\n")
            return

        if save_path is None:
            print()
            default_dir = Path.cwd()
            default_name1 = "traj.json"
            default_name2 = f"traj_{datetime.now():%Y%m%d_%H%M%S}.json"

            dir_input = input(f"Enter directory to save file (default: {default_dir}): ").strip()
            save_dir = Path(dir_input) if dir_input else default_dir

            print("Choose filename:")
            print(f"  [1] {default_name1}")
            print(f"  [2] {default_name2}")
            name_choice = input("Enter filename or 1/2 (default: 1): ").strip()

            if not name_choice or name_choice == "1":
                file_name = default_name1
            elif name_choice == "2":
                file_name = default_name2
            else:
                file_name = name_choice

            save_path = save_dir / file_name

        save_path = Path(save_path).expanduser().resolve()
        save_path.parent.mkdir(parents=True, exist_ok=True)

        with open(save_path, "w") as f:
            json.dump(self._record_list, f, indent=2)

        print(f"\n[OK] Saved trajectory to {save_path}\n")
        self._save_path = save_path  # remember last location

    def load(self, load_path: str | None = None):
        """@brief Load samples from a JSON file.
        @param load_path Optional path; if omitted, prompts for dir/name.
        @note Validates existence and updates current buffer on success.
        """
        if load_path is None:
            print()
            default_dir = Path.cwd()
            default_name = "traj.json"

            dir_input = input(f"Enter directory to load file from (default: {default_dir}): ").strip()
            load_dir = Path(dir_input) if dir_input else default_dir

            name_choice = input(f"Enter filename (default: {default_name}): ").strip()
            file_name = name_choice or default_name

            load_path = load_dir / file_name

        load_path = Path(load_path).expanduser().resolve()

        if not load_path.exists():
            print(f"\n[ERR] File not found: {load_path}\n")
            return

        try:
            with open(load_path, "r") as f:
                self._record_list = json.load(f)
            print(f"\n[OK] Loaded trajectory from {load_path}\n")
            self._save_path = load_path  # remember last location
        except Exception as e:
            print(f"\n[ERR] Failed to load trajectory: {e}\n")

    # ----------------- interactive menu -----------------
    def start_cli(self):
        """@brief Launch a single-key interactive menu.
        @details Keys: q quit | r start | c stop | p play | P loop |
                 s save | l load | f release | d focus.
        """
        self._print_menu()
        while True:
            key = readchar.readkey()

            if key == "q":
                print("\nQuitting…\n")
                break
            elif key == "r":
                self.start_record()
            elif key == "c":
                self.stop_record()
            elif key == "p":
                if self._playing:
                    print("\n[!] A trajectory is currently being replayed.\n")
                    continue
                stride = self._get_stride()
                print(f"\n▶ Starting single playback with stride = {stride}\n")
                self.play(stride)
                self._print_menu()
            elif key == "s":
                self.save()
            elif key == "l":
                self.load()
            elif key == "f":
                self.mc.release_all_servos()
                print("\n[OK] Servos released.\n")
            elif key == "d":
                self.mc.focus_all_servos()
                print("\n[OK] Servos focused.\n")
            else:
                print(f"\n[?] Unknown key: {key!r}\n")
                self._print_menu()

    def _print_menu(self):
        """@brief Print menu help."""
        print(
            """\n
q: quit
r: start recording (will release motors) 
c: stop recording 
p: play trajectory
s: save trajectory 
l: load trajectory 
f: release motors 
d: lock motors
----------------------------------
"""
        )

    def _get_stride(self) -> int:
        """@brief Prompt for stride; return validated value.
        @return Stride ≥ 1, default=1.
        """
        max_stride = max(1, len(self._record_list))
        default = 1
        print()
        try:
            print(f"Choose stride length (1-{max_stride}, default = {default})")
            print(f"A stride length of 3 plays every third point of the trajectory. A higher stride gives a higher playback speed.")
            raw = input(f"Enter stride length:").strip()
            stride = int(raw) if raw else default
        except ValueError:
            print("Invalid number; using default.")
            stride = default
        return stride
    
    def _echo(self, msg):
        print("\r" + msg + " " * 10, end="")

