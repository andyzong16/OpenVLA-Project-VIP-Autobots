# Install minimal dependencies (`torch`, `transformers`, `timm`, `tokenizers`, ...)
# > pip install -r https://raw.githubusercontent.com/openvla/openvla/main/requirements-min.txt
from transformers import AutoModelForVision2Seq, AutoProcessor
from pymycobot import MyCobot280
from PIL import Image
import time
import camera.cv2cam as cam
import cv2
import torch
import numpy as np
from PIL import Image

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
    pil_image = pil_image.resize((224, 224))

    # Grab image input & format prompt
    instruction = "pick up the light green block"
    prompt = f"In: What action should the robot take to {instruction}?\nOut:"

    def adjust_inputs(inputs_dict):
        # Convert to CPU for manipulation
        inputs_cpu = {k: v.cpu() if isinstance(v, torch.Tensor) else v for k, v in inputs_dict.items()}
        
        with torch.no_grad():
            # Initialize a conversion dict that preserves correct dtypes for each input
            convert_dict = {}
            for k, v in inputs_cpu.items():
                if isinstance(v, torch.Tensor):
                    if k in ['input_ids', 'attention_mask']:
                        # embedding indices must be integer types
                        convert_dict[k] = v.to('cuda:0', dtype=torch.long)
                    elif k == 'pixel_values':
                        # image tensor: keep bfloat16 for efficient processing
                        convert_dict[k] = v.to('cuda:0', dtype=torch.bfloat16)
                else:
                    convert_dict[k] = v

            outputs = vla(
                **convert_dict,
                output_projector_features=True,
                return_dict=True,
            )
            
        if outputs.projector_features is not None:
            # Calculate target sequence length
            n_visual_tokens = outputs.projector_features.shape[1] - 1  # Reduce by 1 to match attention size
            
            # Update pixel_values to match (if needed)
            if 'pixel_values' in inputs_cpu:
                # Adjust the spatial dimensions to get desired number of patches
                spatial_size = int(np.sqrt(n_visual_tokens))
                inputs_cpu['pixel_values'] = torch.nn.functional.interpolate(
                    inputs_cpu['pixel_values'],
                    size=(spatial_size * 32, spatial_size * 32),  # Assuming patch size of 32
                    mode='bilinear',
                    align_corners=False
                )
        
        # Move back to GPU with appropriate dtypes
        return {k: (v.to("cuda:0", dtype=torch.bfloat16) if k == 'pixel_values' else
                   v.to("cuda:0", dtype=torch.long) if k in ['input_ids', 'attention_mask'] else
                   v.to("cuda:0") if isinstance(v, torch.Tensor) else v)
                for k, v in inputs_cpu.items()}

    # Predict Action (7-DoF; un-normalize for BridgeData V2)
    inputs = processor(prompt, pil_image)
    inputs = adjust_inputs(inputs)
    
    print("\nAdjusted input shapes:")
    for k, v in inputs.items():
        if isinstance(v, torch.Tensor):
            print(f"{k}: {tuple(v.shape)}")
            if k == 'input_ids':
                print(f"Number of tokens: {v.shape[1]}")

    # Get projector features for verification
    with torch.no_grad():
        outputs = vla(**inputs, output_projector_features=True, return_dict=True)
            
    print(f"\nProjector features shape: {tuple(outputs.projector_features.shape)}")
    print("\nModel outputs shapes:", {k: v.shape for k, v in outputs.items() if isinstance(v, torch.Tensor)})
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
