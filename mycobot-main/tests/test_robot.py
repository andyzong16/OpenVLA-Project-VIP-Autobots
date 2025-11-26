from robot import Robot
from pymycobot import MyCobot280
import camera.OAK as oak
import camera.realsense305 as rs305
import numpy as np

# create robot object
robert = Robot(rs305.RGBD(), mc=MyCobot280)

robert.clickCaptureLoop()
# # calibrate the robot
# robert.calibrate()

# # register the mouse click callback
# robert.registerMouseClicks()

# # loop until q is pressed
# while not robert.closeAllImageWindows('q'):
    
#     # display the image
#     robert.display()
    
#     # get the world positions where the mouse has been clicked
#     worldPos = robert.getMouseWorldPosition('w')
#     if worldPos is not None:
#         print(worldPos, isinstance(worldPos, np.ndarray))
#         robert.captureObject(worldPos[-1])