import manipulator
from pymycobot import MyCobot280
from camera.extrinsic.arucoExtrinsic import calibrate, getExtrinsics
from camera.utils.transformations import pixelToWorkspace, workspaceToPixel
from camera.tabletop.height_estimator import HeightEstimator
import os
import cv2
import numpy as np
from importlib.util import find_spec
import yaml
import time
from enum import Enum
from skimage.measure import regionprops
import readchar

class Robot():
    def __init__(self,
                 eye_cam,
                 mc:MyCobot280 = None,
                 arm_port="/dev/ttyACM0", 
                 arm_baud="115200", 
                 arm_estop=True,
                 hand_speed=20):
        # setup robot
        if mc is None:
            mc = MyCobot280(port=arm_port, baudrate=arm_baud)
        else:
            mc = mc(port=arm_port, baudrate=arm_baud)

        self.manip = manipulator.Manipulator(mc)
        # camera parameters
        self.eye = eye_cam
        # try to get paths and create them if they don't exist yet
        self.eye_extrinsicDir = os.path.dirname(os.path.realpath(__file__))+"/../utils/robotExtrinsics"
        self.eye_arucoTagDir = os.path.dirname(os.path.realpath(__file__))+"/../utils/arucoTag.yaml"
        self.eye_gCW = None
        self.eye_gWC = None
        self.eye_camMTX = None
        self.eye_distCoeffs = None
        self.eye_windowName = None
        self.eye_heightEstimator = None
        self.eye_visualizeTableTopWindowName = None
        self.eye_objectsWindowName = None
        class WindowNames(Enum):
            EYE = 0
            HEIGHTMAP = 1
            OBJECTS = 2
        self.windowNames = WindowNames

        # load base to workspace transformations
        package_root = os.path.dirname(find_spec("mc280").origin)
        transform_path = os.path.join(package_root, "../utils/transforms.yaml")
        with open(transform_path, "r") as file:
            data = yaml.safe_load(file)
        self.gBW = data["T_BW"]
        self.gWB = data["T_WB"]
        self.dhTable = np.asarray(data["dhTable"], dtype=float)

        # start the camera
        self.eye.start()


    def calibrate(self, imageBoarderClipping=0.03):
        '''Fully Calibrates the robot

        Args:
            imageBoarderClipping (optional):
                The percent at which the boarder of all images will be clipped
        '''
        # calibrate the camera of the robot
        self.eye.stop()
        calibrate(self.eye_extrinsicDir, self.eye_arucoTagDir, cam=self.eye)
        self.eye.start()
        
        # store the transformation in data members
        self.eye_gCW, self.eye_gWC, self.eye_camMTX, self.eye_distCoeffs = getExtrinsics(self.eye_extrinsicDir+"/extrinsics.yaml")

        # setup height estimator
        self.eye_heightEstimator = HeightEstimator(self.eye_camMTX)

        # wait some time before geting processing frame
        startTime = time.time()
        while time.time() < (startTime + 3):
            images = self.eye.capture(normalization=False)
        # use frame for depth calibration
        images.depth = images.depth # convert to meters
        # normalize depth to be a max of 1 meter
        mask = images.depth>1
        images.depth[mask] = 1
        self.eye_heightEstimator.calibrate(depth_map=images.depth)

        # store the image boarder clipping percentage
        self.eye_imageBoarderClipping = imageBoarderClipping
        shape = np.array(images.color.shape)
        self._startCroppingPixels = (shape * self.eye_imageBoarderClipping).astype(np.uint32)
        self._endCroppingPixels = (shape - self._startCroppingPixels).astype(np.uint32)


    def getExtrinsics(self):
        '''Returns the extrinsic parameters: gCW, gWC, camera intrinsic matrix, camera distortion coefficients.

        Reads in the extrinsic parameters from the yaml if not already initialized
        Returns:
                gCW , gWC, camMTX, distortionCoeffs
        '''
        # check if we have not already initialized parameters
        if (self.eye_gWC is None) or (self.eye_gCW is None) or (self.eye_camMTX is None) or (self.eye_distCoeffs is None):
            # store the transformation in data members
            self.eye_gCW, self.eye_gWC, self.eye_camMTX, self.eye_distCoeffs = getExtrinsics(self.eye_extrinsicDir+"/extrinsics.yaml")
        return self.eye_gCW, self.eye_gWC, self.eye_camMTX, self.eye_distCoeffs
    

    def display(self):
        '''Displays the latest image captured by the camera'''
        self.eye_windowName = "eyes"
        frames = self.eye.capture(normalization=False)
        # clippedImage = frames.color[self._startCroppingPixels[0]:self._endCroppingPixels[0], self._startCroppingPixels[1]:self._endCroppingPixels[1]]
        self.eye.display(frames.color, self.eye_windowName)


    def closeAllImageWindows(self, key):
        '''Closes all image windows if key is pressed
        Returns:
            True if 'key' has been pressed. 
            False if 'key' has not been pressed
        '''
        opkey = cv2.waitKey(30)
        if opkey == ord(key):
            # check each window is active
            # if self.eye_windowName is not None:
            #     cv2.destroyWindow(self.eye_windowName)
            # if self.eye_visualizeTableTopWindowName is not None:
            #     cv2.destroyWindow(self.eye_visualizeTableTopWindowName)
            # if self.eye_objectsWindowName is not None:
            #     cv2.destroyWindow(self.eye_objectsWindowName)
            cv2.destroyAllWindows()
            return True
        else:
            return False
        
    
    def closeImageWindow(self, key, window):
        '''Closes a specific image window based on the window enum
        Returns:
            True if 'key' has been pressed. 
            False if 'key' has not been pressed
        '''
        opkey = cv2.waitKey(30)
        if opkey == ord(key):
            if window == self.windowNames.EYE:
                # check window is currently active
                if self.eye_windowName is not None:
                    cv2.destroyWindow(self.eye_windowName)
                    self.eye_windowName = None
                    return True
                else:
                    return False
            if window == self.windowNames.HEIGHTMAP:
                # check window is currently active
                if self.eye_visualizeTableTopWindowName is not None:
                    cv2.destroyWindow(self.eye_visualizeTableTopWindowName)
                    self.eye_visualizeTableTopWindowName = None
                    return True
                else:
                    return False
            if window == self.windowNames.OBJECTS:
                # check if window is currently active
                if self.eye_objectsWindowName is not None:
                    cv2.destroyWindow(self.eye_objectsWindowName)
                    self.eye_objectsWindowName = None
                    return True
                else:
                    return False
        else:
            return False
        

    def registerMouseClicks(self):
        '''Sets up window to record mouse clicks'''
        self.eye_windowName = "eyes"
        self.eye.registerMouseClicks(self.eye_windowName)

    
    def getMouseClicks(self, key):
        '''Gets the mouse clicks recorded on a window. After reading the clicks are cleared. Returns None if no clicks have happened or key not pressed
        Args:
            key (char) : key to press to retrieve mouse clicks
        Returns:
            clicks :
                (ndarray(2, N)) - N is the number of clicks.

        '''
        opkey = cv2.waitKey(20)
        if opkey == ord(key):
            return self.eye.readClicks()
        else:
            return None


    def getMouseWorldPosition(self, key):
        '''Gets the world position of a keypress or multiple key presses
        Args:
            key (char) : key to press to retrieve mouse clicks
        Returns:
            worldPositions:
                (ndarray(4, N)) - N is the number of clicks
        '''
        clicks = self.getMouseClicks(key)
        if clicks is not None:
            frames = self.eye.capture(normalization=False)
            # frames.depth = frames.depth[self._startCroppingPixels[0]:self._endCroppingPixels[0], self._startCroppingPixels[1]:self._endCroppingPixels[1]]
            worldPositions = []
            for click in range(0, clicks.shape[1]):
                # print(np.array([[clicks[0][click]],[clicks[1][click]]]))
                newPos = pixelToWorkspace(np.array([[clicks[0][click]],[clicks[1][click]]]), frames.depth, self.eye_gWC, self.eye_camMTX)
                worldPositions.append(newPos)
            return worldPositions
        return None
    

    def getTableTopWorkspace(self):
        '''Gets the height of items above the table top workspace

        Returns:
            heightMap:
                the heigh of objects above the table top workspace
        '''
        images = self.eye.capture(normalization=False)
        depth = np.copy(images.depth)
        # images.depth = images.depth[self._startCroppingPixels[0]:self._endCroppingPixels[0], self._startCroppingPixels[1]:self._endCroppingPixels[1]]
        # images.color = images.color[self._startCroppingPixels[0]:self._endCroppingPixels[0], self._startCroppingPixels[1]:self._endCroppingPixels[1]]
        # save the depth image for possible use in finding objects in workspace
        self._tableTopDepth = images.depth
        self._tableTopColor = images.color
        # convert to meters
        depth = depth/1000
        # normalize depth map to have a max depth of 1 meter
        mask=depth>1
        depth[mask] = 1
        # get height map
        heightMap = self.eye_heightEstimator.apply(depth)
        # heightMap = heightMap[self._startCroppingPixels[0]:self._endCroppingPixels[0], self._startCroppingPixels[1]:self._endCroppingPixels[1]]
        return heightMap
    
    
    def visualizeTableTopWorkspace(self, heightMap):
        '''Displays the table top workspace as a heatmap

        Args:
            heightMap:
                height map returned from getTableTopWorkspace
        '''
        self.eye_visualizeTableTopWindowName = "heatmap of table top workspace"
        imgMin = np.min(heightMap)
        imgMax = np.max(heightMap)
        normalizedheightMap = np.interp(heightMap, (imgMin, imgMax), (0, 255)).astype(np.uint8)
        normalizedheightMap = cv2.applyColorMap(normalizedheightMap, cv2.COLORMAP_DEEPGREEN)
        cv2.imshow(self.eye_visualizeTableTopWindowName, normalizedheightMap)


    def maskTableTopHeight(self, heightMap, threshold):
        '''Returns a mask of all the pixels that are above the given threashold
        Args:
            heightMap:
                heigh map returned from getTableTopWorkspace
            threshold:
                the threshold that objects need to be strictly above to have a value of True.
                The threshold should be given in meters.
        Returns:
            heightMapMask:
                a masked array of the height map where values are either True or False 
                depending on if they are above or below the threshold
        '''
        mask = heightMap > threshold
        return mask
    

    def applyMaskToHeightMap(self, heightMap, mask):
        '''Applies a mask to a height map.
        Args:
            heightMap:
                height map returned from getTableTopWorkspace
            mask:
                mask returned from maskTableTopHeight
        Returns:
            maskedHeightmap:
                a height map that has been masked by the given mask. 
                Values will either be 0 or thier current heightMap value depending on the mask
        '''
        maskedHeightMap = heightMap * mask
        return maskedHeightMap
    


    def findObjectsInWorkspace(self, color:str, maskValues=None, visualization=True):
        '''Returns a ndarray of objects in the workspace
        Args:
            color (str):
                the color of the object you wish to find. Supported colors: 'green', 'red', 'blue', 'custom'.\n
                custom requires the configuration of redMask, blueMask, greenMask to find object
            maskValues (tuple):
                the mask values for red, green, and blue. This is only used for 'custom'.\n
                maskValues = (redGreaterThan, redLessThan, greenGreaterThan, greenLessThan, blueGreaterThan, blueLessThan)
        Returns:
            middleWorldCoords (ndarray (5, N)):
                ndarray of the coordinates of the object found in the world space
        '''
        # get a new set of frames
        images = self.eye.capture(normalization=False)
        # get red objects -> bgr format for realsense305
        red = images.color[:,:,2]
        green = images.color[:,:,1]
        blue = images.color[:,:,0]
        # use color as mask input
        if color != "custom":
            # apply red mask
            if color == "red":
                mask = (red > 100)*(green < 60)*(blue < 60)
            # apply green mask
            if color == "green":
                mask = (red < 60)*(green > 100)*(blue < 60)
            # apply blue mask
            if color == "blue":
                mask = (red < 60)*(green < 60)*(blue > 100)
        # use maskvalues as mask input
        else:
            if maskValues is not None:
                redGreaterThan = maskValues[0]
                redLessThan = maskValues[1]
                greenGreaterThan = maskValues[2]
                greenLessThan = maskValues[3]
                blueGreaterThan = maskValues[4]
                blueLessThan = maskValues[5]
                mask = (red > redGreaterThan)*(red < redLessThan)*(green > greenGreaterThan)*(green < greenLessThan)*(blue > blueGreaterThan)*(blue < blueLessThan)
            else:
                print("maskValues cannont be 'None' for color=custom")
                return None
            
        mask[0:3,:] = 0
        mask = mask.astype(np.uint8)
        regions = regionprops(mask)
        for region in regions:
            box = region.bbox
            middle = np.array([[int(box[1] + 0.5*(box[3] - box[1]))],[int(box[0] + 0.5*(box[2]-box[0]))]])
            middleWorldCoords = pixelToWorkspace(middle, images.depth, self.eye_gWC, self.eye_camMTX)
            if visualization:
                cv2.rectangle(images.color, (box[1], box[0]), (box[3], box[2]), (225, 0, 0), 2)
                
        if visualization:
            self.eye_objectsWindowName = "visualize objects in workspace"
            cv2.imshow(self.eye_objectsWindowName, images.color)
            images.color[:,:,0] = images.color[:,:,0]*mask
            images.color[:,:,1] = images.color[:,:,1]*mask
            images.color[:,:,2] = images.color[:,:,2]*mask
        return middleWorldCoords




        # get height map and mask
        heightMap = self.getTableTopWorkspace()
        mask = self.maskTableTopHeight(heightMap, threshold)
        # find all regions of objects satsifying the threshold
        regions = regionprops((heightMap * mask * 50).astype(np.uint8))
        # go through all regions and add objects to the array
        objects = np.array([[],[]])
        for region in regions:
            box = region.bbox
            # topLeft = pixelToWorkspace(np.array([[box[1]],[box[0]]]), self._tableTopDepth, self.eye_gWC, self.eye_camMTX)
            # bottomRight = pixelToWorkspace(np.array([[box[3]-1],[box[2]-1]]), self._tableTopDepth, self.eye_gWC, self.eye_camMTX)
            # np.append(objects, np.array([[topLeft], [bottomRight]]))
            if visualization:
                cv2.rectangle(self._tableTopColor, (box[0], box[2]), (box[1], box[3]), (255, 0, 0))
                
        if visualization:
            cv2.imshow(self.eye_objectsWindowName, self._tableTopColor)
        return objects



    def captureObject(self, coords, hover_height=50):
        '''Captures an object in the workspace
        Args:
            coords (ndarray): the coordinates of an object in the workspace frame in meters
        '''
        coords = coords*1000 # convert to mm for capture function
        xyz = coords.flatten().tolist()
        self.manip.capture(xyz, hover_height=hover_height)

    def releaseObject(self, coords, release_height=50):
        coords = coords*1000 # convert to mm for capture function
        xyz = coords.flatten().tolist()
        self.manip.release(xyz, release_height=release_height)

    def waitForSingleClickWorld(self):
        """
        @brief Blocks until a single mouse click is registered and returns its
        position in the workspace frame.

        Uses the current depth map and extrinsics to convert the last click
        to world coordinates.

        @return coords (ndarray): coordinates of the clicked point in the
            workspace frame in meters.
        """
        # Flush stale clicks
        while True:
                old_clicks = self.eye.readClicks()
                if old_clicks is None or old_clicks.size == 0:
                    break

        while True:
            # Keep the camera image updating so the window stays responsive
            self.display()

            # Read and clear any stored clicks from the camera object
            clicks = self.eye.readClicks()
            if clicks is not None and clicks.size > 0:
                # Use the last click
                u = int(clicks[0, -1])
                v = int(clicks[1, -1])

                # Grab a depth frame for this click
                frames = self.eye.capture(normalization=False)
                coords = pixelToWorkspace(
                    np.array([[u], [v]]),
                    frames.depth,
                    self.eye_gWC,
                    self.eye_camMTX,
                )
                return coords

            # Let OpenCV process events; small delay to avoid busy spinning
            cv2.waitKey(10)


    def clickCaptureLoop(self, capture_key="c", release_key="r", quit_key="q"):
        """
        Before calling function, ensure that the camera gets a clear view of the aruco tag.
        Interactive loop:
            - Press capture_key, then click a point to capture at that position.
            - Press release_key, then click a point to release at that position.
            - Press quit_key to exit.

        Assumes waitForSingleClickWorld() returns coordinates in workspace
        frame in meters, suitable for captureObject(..).
        """

        self.calibrate()
        self.registerMouseClicks()

        print(f"Press '{capture_key}' to capture at a clicked point.")
        print(f"Press '{release_key}' to release at a clicked point.")
        print(f"Press '{quit_key}' to quit.")
        print(f"Press 'h' to move arm to home position.")
        print(f"Press 'o/p' to open or close the gripper")

        while True:
            # Keep updating the camera window
            self.display()

            key = cv2.waitKey(20) & 0xFF

            if key == ord(quit_key):
                cv2.destroyAllWindows()
                break

            elif key == ord(capture_key):
                print("Capture mode: click on the object to capture...")
                coords = self.waitForSingleClickWorld().flatten()
                print("Capturing object at coordinate: ", coords)

                try:
                    self.captureObject(coords)
                except Exception as e:
                    print(f"An error occurred: {e}")

            elif key == ord(release_key):
                print("Release mode: click on two points to release above the midpoint...")
                release_height = 50

                try:
                    user_in = input("Enter release hover height in mm (press enter to keep default=50): ").strip()
                    if user_in:
                        release_height = float(user_in)
                except:
                    print("Invalid input, keeping default.")

                coords1 = self.waitForSingleClickWorld().flatten()
                print("Got first release point: ", coords1)

                coords2 = self.waitForSingleClickWorld().flatten()
                print("Got second release point: ", coords2)

                midpoint = 0.5 * (coords1 + coords2)
                
                print("Releasing in the midpoint: ", midpoint)

                try:
                    self.releaseObject(midpoint, release_height)
                except Exception as e:
                    print(f"An error occurred: {e}")
                

            elif key == ord("h"):
                print("Moving to home position.")
                self.manip.arm.goto_home()

            elif key == ord("o"):
                print("Opening gripper.")
                self.manip.hand.set(self.manip.hand.cfg.command_range[1])
            elif key == ord("p"):
                print("Closing gripper.")
                self.manip.hand.set(self.manip.hand.cfg.command_range[0])