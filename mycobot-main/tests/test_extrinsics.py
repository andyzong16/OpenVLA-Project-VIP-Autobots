''' Test of the extrinsic calculations for camera to workspace

You must run 'arucoCalibration.py' in directory 'extrinsics' before running this file

Goal: Draw a box around an object give it's coordinates in the world/workspace frame

Author: Kyle de Nobel

'''

import camera.OAK as oak
import yaml
import os
import numpy as np
import cv2
import ivapy.display_cv as display
# from utils.arucoExtrinsics import getExtrinsics


# ==================== SETUP ==================== #

# calibrate


# extract information from yaml file
currentDir = os.path.dirname(os.path.realpath(__file__))
with open(currentDir+"/../utils/mycobotExtrinsics/extrinsics.yaml", 'r') as stream:
    extrinsics = yaml.load(stream, Loader=yaml.FullLoader)

gCW = extrinsics['gCW']
camMTX = extrinsics['camMTX']
distortionCoeffs = extrinsics['distortionCoeffs']

# gCW, _, camMTX, distortionCoeffs = getExtrinsics()

# create camera object
cam = oak.RGBD()

# get coordinate system axes
gCW = np.array(gCW)
rvecs, _ = cv2.Rodrigues(gCW[:3,:3])
tvecs = np.reshape(gCW[:3,3], (3,1))

camMTX = cv2.Mat(np.array(camMTX))
distortionCoeffs = cv2.Mat(np.array(distortionCoeffs))



# ==================== Drawing initial square on image ==================== #



# define the top left and top right coordinates of block in workspace
blockSize = 0.03 # meters
blockBottomRightX = 0.0
blockBottomRightY = 0.06
blockBottomRight = np.array([[blockBottomRightX],[blockBottomRightY],[0], [1]]) # bottom right corner
blockBottomLeft = np.array([[blockBottomRightX+blockSize], [blockBottomRightY], [0], [1]]) # bottom left corner
blockTopRight = np.array([[blockBottomRightX], [blockBottomRightY+blockSize], [0], [1]]) # top right corner
blockTopLeft = np.array([[blockBottomRightX+blockSize],  [blockBottomRightY+blockSize],[0],[1]]) # top left corner

# perform caluclation to draw box around block
# bottom right
blockInCameraFrame = gCW@blockBottomRight
blockInCameraFrame = blockInCameraFrame[:3]    # do not want the 1 appended to vector
blockInPixelFrame = camMTX@blockInCameraFrame
u1 = blockInPixelFrame[0]/blockInPixelFrame[2] # u is horizontal coordinate in image
v1 = blockInPixelFrame[1]/blockInPixelFrame[2] # v is vertical coordinate in image

# bottom left
blockInCameraFrame = gCW@blockBottomLeft
blockInCameraFrame = blockInCameraFrame[:3]    # do not want the 1 appended to vector
blockInPixelFrame = camMTX@blockInCameraFrame
u2 = blockInPixelFrame[0]/blockInPixelFrame[2] # u is horizontal coordinate in image
v2 = blockInPixelFrame[1]/blockInPixelFrame[2] # v is vertical coordinate in image

# top right
blockInCameraFrame = gCW@blockTopRight
blockInCameraFrame = blockInCameraFrame[:3]    # do not want the 1 appended to vector
blockInPixelFrame = camMTX@blockInCameraFrame
u3 = blockInPixelFrame[0]/blockInPixelFrame[2] # u is horizontal coordinate in image
v3 = blockInPixelFrame[1]/blockInPixelFrame[2] # v is vertical coordinate in image

# top left
blockInCameraFrame = gCW@blockTopLeft
blockInCameraFrame = blockInCameraFrame[:3]    # do not want the 1 appended to vector
blockInPixelFrame = camMTX@blockInCameraFrame
u4 = blockInPixelFrame[0]/blockInPixelFrame[2] # u is horizontal coordinate in image
v4 = blockInPixelFrame[1]/blockInPixelFrame[2] # v is vertical coordinate in image




# ==================== Capture image and perform drawing update ==================== #



# start camera
cam.start()

# let user know instructions
print("="*50)
print("press 'i' to input new x and y coordinates for the block")
print("the coordinates must be in meters and separated by a single space: x y")
print("press 'q' to quit the program")
print("="*50)

while True:
    # get new image
    images = cam.capture()
    
    # draw axes on image
    cv2.drawFrameAxes(images.color, camMTX, distortionCoeffs, rvecs, tvecs, 0.07*1.5, 2)


    # draw lines on the image. Need [0] after every variable since u1 = list of one ouble
    # print(u1) -> [double]
    cv2.line(images.color, (int(u1[0]), int(v1[0])), (int(u2[0]), int(v2[0])), (255,0,0), 2)
    cv2.line(images.color, (int(u1[0]), int(v1[0])), (int(u3[0]), int(v3[0])), (255,0,0), 2)
    cv2.line(images.color, (int(u4[0]), int(v4[0])), (int(u2[0]), int(v2[0])), (255,0,0), 2)
    cv2.line(images.color, (int(u4[0]), int(v4[0])), (int(u3[0]), int(v3[0])), (255,0,0), 2)


    # display image with rectange around block
    cam.display(images.color, "outlined block")

    opKey = display.wait(1)
    if opKey == ord('q'):
        break

    # if user presses i, take new input and redraw box
    if opKey == ord('i'):
        userInput = input()
        x = float(userInput.split(" ")[0])
        y = float(userInput.split(" ")[1])
        blockBottomRightX = x
        blockBottomRightY = y
        # calculate corners of the block
        blockBottomRight = np.array([[blockBottomRightX],[blockBottomRightY],[0], [1]]) # bottom right corner
        blockBottomLeft = np.array([[blockBottomRightX+blockSize], [blockBottomRightY], [0], [1]]) # bottom left corner
        blockTopRight = np.array([[blockBottomRightX], [blockBottomRightY+blockSize], [0], [1]]) # top right corner
        blockTopLeft = np.array([[blockBottomRightX+blockSize],  [blockBottomRightY+blockSize],[0],[1]]) # top left corner

        #  ======  block  ======  #
        #                         #
        #   (u4, v4)    (u3, v3)  #
        #                         #
        #   (u2, v2)    (u1,v1)   #
        #                         #
        #  =====================  #

        # perform caluclation to draw box around block
        # bottom right
        blockInCameraFrame = gCW@blockBottomRight
        blockInCameraFrame = blockInCameraFrame[:3]    # do not want the 1 appended to vector
        blockInPixelFrame = camMTX@blockInCameraFrame
        u1 = blockInPixelFrame[0]/blockInPixelFrame[2] # u is horizontal coordinate in image
        v1 = blockInPixelFrame[1]/blockInPixelFrame[2] # v is vertical coordinate in image

        # bottom left
        blockInCameraFrame = gCW@blockBottomLeft
        blockInCameraFrame = blockInCameraFrame[:3]    # do not want the 1 appended to vector
        blockInPixelFrame = camMTX@blockInCameraFrame
        u2 = blockInPixelFrame[0]/blockInPixelFrame[2] # u is horizontal coordinate in image
        v2 = blockInPixelFrame[1]/blockInPixelFrame[2] # v is vertical coordinate in image

        # top right
        blockInCameraFrame = gCW@blockTopRight
        blockInCameraFrame = blockInCameraFrame[:3]    # do not want the 1 appended to vector
        blockInPixelFrame = camMTX@blockInCameraFrame
        u3 = blockInPixelFrame[0]/blockInPixelFrame[2] # u is horizontal coordinate in image = u/w
        v3 = blockInPixelFrame[1]/blockInPixelFrame[2] # v is vertical coordinate in image   = v/w

        # top left
        blockInCameraFrame = gCW@blockTopLeft
        blockInCameraFrame = blockInCameraFrame[:3]    # do not want the 1 appended to vector
        blockInPixelFrame = camMTX@blockInCameraFrame
        u4 = blockInPixelFrame[0]/blockInPixelFrame[2] # u is horizontal coordinate in image
        v4 = blockInPixelFrame[1]/blockInPixelFrame[2] # v is vertical coordinate in image


        # draw lines on the image. Need [0] after every variable since u1 = list of one ouble
        # print(u1) -> [double]
        cv2.line(images.color, (int(u1[0]), int(v1[0])), (int(u2[0]), int(v2[0])), (255,0,0), 2)
        cv2.line(images.color, (int(u1[0]), int(v1[0])), (int(u3[0]), int(v3[0])), (255,0,0), 2)
        cv2.line(images.color, (int(u4[0]), int(v4[0])), (int(u2[0]), int(v2[0])), (255,0,0), 2)
        cv2.line(images.color, (int(u4[0]), int(v4[0])), (int(u3[0]), int(v3[0])), (255,0,0), 2)

        print("u = ", u1, "  |  v = ", v1)

