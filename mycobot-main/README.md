# ⚠️ Common Issues
### [Linux] Permission denied when connecting to the robot
If you see an error like:

`PermissionError: [Errno 13] Permission denied: '/dev/ttyACM0'`

that means your Linux user does not have permission to access the serial port that MyCobot280 is connected to by default (`/dev/ttyACM0`).

**Fix: Add your user to the `dialout` group (which grants access to serial devices)**

```bash
sudo usermod -a -G dialout $USER
```

Then log out and log back in for the change to take effect. If this doesn't resolve the issue, try powering off the device and the turn it back on.

### [Windows] PyBullet installation failure
If you see an error like:

`error: Microsoft Visual C++ 14.0 or greater is required`

this means `pip` is trying to **compile PyBullet from source**, but Windows doesn’t come with a C++ compiler by default. 

**Fix Option 1 (Recommended): Use Conda Prebuilt Binaries**
```bash
conda install -c conda-forge pybullet
```
This will install the precompiled packages for PyBullet if you are using a Conda environment.

**Fix Option 2: Install Microsoft Build Tools**

1. Download from: https://visualstudio.microsoft.com/visual-cpp-build-tools/

2. Run the installer and select
- Desktop development with C++
- Include: MSVC v143 build tools and Windows 10/11 SDK

3. Restart your terminal, then run:
```bash
pip install pybullet
```

### [Hardware] Arm becomes unresponsive 

If the myCobot 280 suddenly stops responding to motion commands and the top display shows a persistent blue screen (signaling error), consider trying the following fixes:

1. **Power-cycle and reconnect**
   - Unplug the robot’s power supply and USB cable.
   - Wait a few seconds, then reconnect everything and try again.
   - If the issue persists, proceed to reflashing the Atom firmware.

2. **Reflash the Atom controller**
   - Download and install **[myStudio](https://www.elephantrobotics.com/en/mystudio/)** from Elephant Robotics.
   - Connect your PC to the **USB port on the last link** of the arm (the same module with the blue display).
   - In myStudio, select the correct robot model (e.g., *myCobot 280*) and **flash** the Atom firmware. Refer to the official guide on how to do so: **[Atom Firmware](https://docs.elephantrobotics.com/docs/gitbook-en/4-BasicApplication/4.1-myStudio/4.1.2-myStudio_flash_firmwares.html)**
   - Once flashing completes, disconnect and reconnect power to reboot the controller.

This process often resolves cases where the robot becomes unresponsive or displays a persistent blue screen.

### [Hardware] Controller Error: “Joint N exceeds limit position”
Sometimes motion commands repeatedly fail with errors such as:

`Exception: Joint 3 exceeds limit position`

even though the commanded pose is physically within limits. This error usually happens when the servo encoder readings drift from the controller’s internal joint limits, causing the robot to think a joint is out of range even when it isn’t.
**Fix:**
1. Call `set_encoder([2048]*6)` to recenter all encoders to the midpoint of their valid range.
