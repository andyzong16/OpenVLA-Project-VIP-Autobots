from pymycobot import MyCobot280
import time

mc = MyCobot280("/dev/ttyACM0")

mc.power_on()
while not mc.is_power_on():
    time.sleep(0.1)

mc.release_all_servos()
