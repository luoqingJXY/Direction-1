"""控制台与乙电脑之间的工程控制字节。"""

from enum import IntEnum


class ControlCommand(IntEnum):
    START_TEACHING = 1
    STOP_TEACHING = 2
    CLOSE_AFTER_SLEEP = 3


def encode_control(command: ControlCommand) -> bytes:
    return bytes((int(command),))


def decode_control(payload: bytes) -> ControlCommand:
    if len(payload) != 1:
        raise ValueError("控制信息必须恰好一个字节")
    return ControlCommand(payload[0])

