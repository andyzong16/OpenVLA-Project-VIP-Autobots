import os, yaml
import mc280
import numpy as np
from ivapy.Configuration import AlgConfig
from pymycobot import MyCobot280
from yacs.config import CfgNode
from importlib.util import find_spec
from Lie.group import SE3   


def load_cfg():
    pkg_root = os.path.dirname(find_spec("mc280").origin)
    gripper_path = os.path.join(pkg_root, "../utils/gripperConfigs.yaml")
    with open(gripper_path, "r") as file:
        config_dict = yaml.safe_load(file)
    config = AlgConfig(config_dict)
    return config

def build_gripper(mc: MyCobot280, cfg: CfgNode):
    if cfg.selected == "adaptive":
        return mc280.Adaptive(mc, cfg.adaptive)
    elif cfg.selected == "flexible":
        return mc280.Flexible(mc, cfg.flexible)
    elif cfg.selected == "suction":
        return mc280.Suction(mc, cfg.flexible)
    raise Exception("Unrecognized gripper type")

def pose_to_SE3(pose: list):
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

def SE3_to_pose(se3: SE3.Homog):
    """
    @brief Convert a list of coordinates to SE3.Homog.
    @param se3 SE3.Homog pose in the base frame in radians.
    @return [x_mm, y_mm, z_mm, rx_deg, ry_deg, rz_deg].
    """
    x_mm, y_mm, z_mm = se3.getTranslation()
    rx_deg, ry_deg, rz_deg = np.rad2deg(se3.getRPY())
    return [x_mm, y_mm, z_mm, rx_deg, ry_deg, rz_deg]

def SE3_to_matrix(se3: SE3.Homog):
    """
    @brief Convert a list of coordinates to SE3.Homog.
    @return [x_mm, y_mm, z_mm, rx_deg, ry_deg, rz_deg].
    """
    matrix = np.eye(4)
    matrix[0:3,0:3] = se3.getRotation()
    matrix[0:3,3] = se3.getTranslation()[0:3,0]
    return matrix