"""
Copyright 2018 ikeyasu.com
LICENSE: MIT License
"""
import argparse
import json
import urllib.request

import time

# Servo chip of SparkFun Pi Servo Hut
CHIP_ADDRESS = 0x40  # I2C address of the PWM chip.

# Channel addresses
# see https://learn.sparkfun.com/tutorials/pi-servo-hat-hookup-guide#software---python
CHANNEL0_START_ADDRESS = 0x06
CHANNEL0_END_ADDRESS = 0x08
CHANNEL0_ADDRESS_INTERVAL = 4

# Servo rotation
# See https://learn.sparkfun.com/tutorials/pi-servo-hat-hookup-guide#software---python
# and http://akizukidenshi.com/download/ds/towerpro/SG90_a.pdf
ROTATION_STEP = 5000.0 / 4095.0
DEGREE_0 = int(1.45 * 1000 / ROTATION_STEP)  # 1.45 ms
DEGREE_n90 = int(0.5 * 1000 / ROTATION_STEP)  # 0.5ms
DEGREE_90 = int(2.4 * 1000 / ROTATION_STEP)  # 2.4ms
DEGREE_STEP_MIN = 0.5
DEGREE_STEP = (1.45 - DEGREE_STEP_MIN) / 90.0

# Etc
SERVO_COUNT = 8
DEBUG = True


def _dp(msg):
    if DEBUG:
        print(msg)

# noinspection PyMethodMayBeStatic
class ServoDebug:
    def __init__(self, servo_count: int):
        _dp("Servo init:servo_count={}".format(servo_count))
        pass

    def rotate(self, channel: int, degree: float):
        _dp("Servo rotate: channel={}, degree={}".format(channel, degree))


class Servo:

    def __init__(self, servo_count: int):
        _dp("Servo init:servo_count={}".format(servo_count))
        import smbus
        self.bus = smbus.SMBus(1)  # the chip is on bus 1 of the available I2C buses
        self.bus.write_byte_data(CHIP_ADDRESS, 0, 0x20)  # enable the chip
        self.bus.write_byte_data(CHIP_ADDRESS, 0xfe, 0x1e)  # configure the chip for multi-byte write
        for i in range(0, servo_count):
            self._init_servo(i)
        time.sleep(1.5)  # wait for rotating to 0 degree

    def _init_servo(self, channel: int):
        start_address = CHANNEL0_START_ADDRESS + CHANNEL0_ADDRESS_INTERVAL * channel
        end_address = CHANNEL0_END_ADDRESS + CHANNEL0_ADDRESS_INTERVAL * channel

        self.bus.write_word_data(CHIP_ADDRESS, start_address, 0)
        self.bus.write_word_data(CHIP_ADDRESS, end_address, DEGREE_0)

    def rotate(self, channel: int, degree: float):
        start_address = CHANNEL0_START_ADDRESS + CHANNEL0_ADDRESS_INTERVAL * channel
        end_address = CHANNEL0_END_ADDRESS + CHANNEL0_ADDRESS_INTERVAL * channel
        degree_pwm = int((float(degree) * DEGREE_STEP + DEGREE_STEP_MIN) * 1000) + DEGREE_0

        self.bus.write_word_data(CHIP_ADDRESS, start_address, 0)
        self.bus.write_word_data(CHIP_ADDRESS, end_address, degree_pwm)
        _dp("Servo rotate: channel={}, degree={}, start_address={}, end_address={}, degree_pwm={}"
            .format(channel, degree, start_address, end_address, degree_pwm))

def _get(address, port):
    url = "http://{}:{}".format(address, port)
    req = urllib.request.Request(url)
    with urllib.request.urlopen(req) as res:
        return json.loads(res.read().decode())


def _loop(servo, address, port):
    while True:
        actions = _get(address, port)
        if actions is None:
            break
        for channel, action in enumerate(actions):
            servo.rotate(channel, action * 40.0)
        time.sleep(0.5)


def main(parser=argparse.ArgumentParser()):
    import logging
    logging.basicConfig(level=logging.WARN)

    parser.add_argument('--server-address', type=str, help="Server setting")
    parser.add_argument('--server-port', type=int, default=8080, help="Server setting")
    parser.add_argument('--local-debug', action='store_true')
    parser.add_argument('--reset-servo', action='store_true', help="Reset servo position to 0 degree and exit")
    args = parser.parse_args()
    servo = Servo(servo_count=SERVO_COUNT) if not args.local_debug else ServoDebug(servo_count=SERVO_COUNT)

    if not args.reset_servo:
        _loop(servo, args.server_address, args.server_port)


if __name__ == '__main__':
    main()
