import numpy as np
import mc280
import pymycobot

mc = pymycobot.MyCobot280("/dev/ttyACM0")
arm = mc280.MC280(mc)
# ga = mc280.Adaptive(mc)

# --- load transform gFG ---
# gFG = np.array([
#     [0.70710678,  0.70710678, 0,   0.0],
#     [-0.70710678, 0.70710678, 0,   0.0],
#     [0,           0,          1,   1.0],
#     [0,           0,          0,   1.0]
# ])
gFG = np.array([
    [ 0.70710678,  0.0,        -0.70710678,  0.0],
    [-0.70710678,  0.0,        -0.70710678,  0.0],
    [ 0.0,         1.0,         0.0,         0.0],
    [ 0.0,         0.0,         0.0,         1.0],
])


# --- define the point in G ---
pG = np.array([0, 116, -12, 1])

# --- transform G → F ---
pF = gFG @ pG
# --- forward kinematics + rotation ---
se3 = arm.fk(arm.measure_joints())
R = se3.getRotation()

# --- rotate pF (only xyz) into end-effector frame ---
pF_rotated = (R @ pF[0:3])
ex, ey, ez = R[:,0], R[:,1], R[:,2]
print("pF =", pF)
print("R @ pF =", pF_rotated)
print("R @ pF + t=", pF_rotated + se3.getTranslation()[0:3,0])
