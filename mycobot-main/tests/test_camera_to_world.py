'''@brief test camera to world transformation to ensure we can map pixes to world frame
@author Kyle de Nobel
'''

# from utils.arucoExtrinsics import calibrate, getExtrinsics
import camera.OAK as oak
import cv2
import numpy as np
import os
import yaml

# get camera
cam = oak.RGBD()

# can't do the following because utils is not recognized :(
# # perform calibration
# calibrate(cam=cam)

# # get important extrinsics
# gCW, gWC, camMTX, _ = getExtrinsics()

# extract information from yaml file
currentDir = os.path.dirname(os.path.realpath(__file__))
with open(currentDir+"/../utils/robotExtrinsics/extrinsics.yaml", 'r') as stream:
    extrinsics = yaml.load(stream, Loader=yaml.FullLoader)

gCW = extrinsics['gCW']
gWC = extrinsics['gWC']
camMTX = extrinsics['camMTX']
distortionCoeffs = extrinsics['distortionCoeffs']

invCamMTX = np.linalg.inv(camMTX)

# start the camera and visulaize the frames
cam.start()

while True:
    # capture and display images
    color, depth = cam.get_frames()
    cam.display(color, "image frame")

    # break from the loop
    opkey = cv2.waitKey(1)
    if opkey == ord('q'):
        break


    # give pixel coordinates and print out world coordinates
    if opkey == ord('i'):
        coords = input()
        pixelX = int(coords.split(" ")[0])  # this is u
        pixelY = int(coords.split(" ")[1])  # this is v

        #pixel_position = np.array([[x],[y],[1]])

        # grab z in camera frame
        zFromDepth = depth[pixelY][pixelX]
        # make in units of meters
        zFromDepth = zFromDepth/1000

        # calculate u, v, w   array with the following equation
        # u = pixelX * (z coord from depth frame -> ie. value of depth frame at the pixel location)
        # v = pixelY * (z coord from depth frame -> ie. value of depth frame at the pixel location)
        imageFrame = np.array([[pixelX*zFromDepth], [pixelY*zFromDepth],[ zFromDepth]])

        # calculate the camera frame position
        cameraFrame = np.linalg.inv(camMTX) @ imageFrame

        gCW = np.array(gCW)

        # calculate the world frame position
        worldPosition = np.linalg.inv(gCW) @ np.block([[cameraFrame], [1]])
        print(worldPosition)


        
    
