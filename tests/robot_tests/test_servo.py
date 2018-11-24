import unittest
from unittest.mock import MagicMock, call

from robot.robot import Servo
import robot.robot as robot


class MockBus:
    def __init__(self):
        self.write_word_data = MagicMock(name="write_word_data")
        self.write_byte_data = MagicMock(name="write_byte_data")


class TestServo(unittest.TestCase):

    def test_init(self):
        mock_bus = MockBus()
        Servo(8, mock_bus)
        mock_bus.write_byte_data.assert_has_calls(
            [call(robot.CHIP_ADDRESS, 0, 0x20), call(robot.CHIP_ADDRESS, 0xfe, 0x1e)])
        calls = sum(
            [[call(robot.CHIP_ADDRESS, (0x06 + i * 4), 0), call(robot.CHIP_ADDRESS, (0x08 + i * 4), robot.DEGREE_0)] for
             i in range(0, 8)], [])
        mock_bus.write_word_data.assert_has_calls(calls, any_order=True)

    def test_rotate(self):
        mock_bus = MockBus()
        servo = Servo(8, mock_bus)
        servo.rotate(0, 0)
        mock_bus.write_word_data.assert_any_call(robot.CHIP_ADDRESS, (0x08 + 0 * 4), robot.DEGREE_0)
        servo.rotate(0, 90)
        mock_bus.write_word_data.assert_any_call(robot.CHIP_ADDRESS, (0x08 + 0 * 4), robot.DEGREE_90)
        servo.rotate(0, -90)
        mock_bus.write_word_data.assert_any_call(robot.CHIP_ADDRESS, (0x08 + 0 * 4), robot.DEGREE_n90)

if __name__ == '__main__':
    unittest.main()
