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
import numpy as np

# Creates an instance of the robot
mc = MyCobot280("/dev/ttyACM0", 115200)

# Load Processor & VLA
processor = AutoProcessor.from_pretrained("openvla/openvla-7b", trust_remote_code=True)
vla = AutoModelForVision2Seq.from_pretrained(
    "openvla/openvla-7b", 
    attn_implementation="eager",  # [Optional] Requires `flash_attn`
    torch_dtype=torch.bfloat16, 
    low_cpu_mem_usage=True, 
    trust_remote_code=True
).to("cuda:0", dtype = torch.bfloat16)

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
    instruction = "pick up the light green block"
    prompt = f"In: What action should the robot take to {instruction}?\nOut:"

    # Predict Action (7-DoF; un-normalize for BridgeData V2)
    inputs = processor(prompt, pil_image).to("cuda:0", dtype=torch.bfloat16)
    with torch.no_grad():
        outputs = vla(**inputs)
        print("Model outputs shapes:", {k: v.shape for k, v in outputs.items() if isinstance(v, torch.Tensor)})
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

    # Use first 6 joint actions to move the robot
    joint_angles = action[:6]
    mc.send_angles(joint_angles, speed=50)

    ok = cv2.waitKey(1)
    if (ok == ord('q')):
        break
