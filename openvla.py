# Install minimal dependencies (`torch`, `transformers`, `timm`, `tokenizers`, ...)
# > pip install -r https://raw.githubusercontent.com/openvla/openvla/main/requirements-min.txt
from transformers import AutoModelForVision2Seq, AutoProcessor
from pymycobot import MyCobot280
from PIL import Image
import time
import camera.cv2cam as cam
import cv2
import torch
from PIL import Image
# import manipulator as manip
import numpy as np

# Creates an instance of the robot
mc = MyCobot280("/dev/ttyACM0", 115200)
# mn = manip.Manipulator(mc)

# Load Processor & VLA
processor = AutoProcessor.from_pretrained("openvla/openvla-7b", trust_remote_code=True)
vla = AutoModelForVision2Seq.from_pretrained(
    "openvla/openvla-7b", 
    attn_implementation="eager",  # [Optional] Requires `flash_attn`
    torch_dtype=torch.bfloat16, 
    low_cpu_mem_usage=True, 
    trust_remote_code=True,
    revision="main"
).to("cuda:0")

# Initialize Camera
camera = cam.Color()
camera.start()

# Main Loop
while True:

    image, isGood = camera.capture()

    if isGood:
        camera.display(image, window_name = 'image')

    # Convert to PIL & resize
    pil_image = Image.fromarray(image)

    # Grab image input & format prompt
    instruction = "pick up the red block"
    prompt = f"In: What action should the robot take to {instruction}?\nOut: "

    # Predict Action (7-DoF; un-normalize for BridgeData V2)
    inputs = processor(prompt, pil_image).to("cuda:0", dtype=torch.bfloat16)

    action = vla.predict_action(**inputs, unnorm_key="bridge_orig", do_sample=False)

    # Turn the robot arm on
    if mc is None:
        mc.power_on()
        while not mc.is_power_on():
            time.sleep(0.1)

    # Process action output
    if isinstance(action, torch.Tensor):
        action = action.cpu().numpy().tolist()

    # Handle batch dimension
    if isinstance(action[0], (list, torch.Tensor)):
        action = action[0]

    # Grab all vector values outputted by OpenVLA
    OutputVector = np.ndarray.tolist(action[:])

    delta = np.array(OutputVector[:3], dtype=float) * 1000 # Convert to meters
    delta = np.append(delta, np.array(OutputVector[3:6], dtype=float) * (180.0 / np.pi)) # Radians to degrees
    print(delta)

    # Translation to MyCobot280 joint angles and send to robot
    for i in range(6):
        mc.jog_increment_coord(i+1, delta[i], speed=20)
        time.sleep(0.5)

    # Gripper control
    mc.set_gripper_value(int(OutputVector[6] * 100), 80)

    # Exit on 'q' key press
    ok = cv2.waitKey(1)
    if (ok == ord('q')):
        break
