from robot import Robot
from pymycobot import MyCobot280
import camera.OAK as oak
import camera.realsense305 as rs305
import cv2


# create robot object
robert = Robot(rs305.RGBD(), mc=MyCobot280)

# calibrate the robot
robert.calibrate()

# register the mouse callback
robert.registerMouseClicks()

# loop until q is pressed and then close all windows
while not robert.closeAllImageWindows('q'):
    # display the image
    robert.display()

    # press f to find objects. (Might need to press and hold)
    opkey = cv2.waitKey(10)
    if opkey == ord('f'):

        # find the green block
        # worldSpace = robert.findObjectsInWorkspace('custom', (0, 115, 120, 225, 0, 90))


        # find the red block
        worldSpace = robert.findObjectsInWorkspace("red")
        # print out the coordinates in the world frame
        print(worldSpace)