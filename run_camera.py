import cv2
import camera.cv2cam as cam

camera = cam.Color()
camera.start()

while True:
    image, isGood = camera.capture()

    if isGood:
        camera.display(image, window_name = 'image')

    ok = cv2.waitKey(1)

    if (ok == ord('q')):
        break

camera.stop()
cv2.destroyAllWindows()