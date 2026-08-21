"""局域网二进制传递帧。

帧中没有生命时间、时间戳、目标、标签或答案。通道编号、长度和校验只
用于把字节可靠送到对应的乙电脑组件，解帧后不会进入脑。
"""

from __future__ import annotations

import hashlib
import hmac
import socket
import struct

from .entry_points import Channel
from .errors import ProtocolError


MAGIC = b"ALB2"
HEADER = struct.Struct("!4sBI")
SIGNATURE_SIZE = hashlib.sha256().digest_size
DEFAULT_MAX_PAYLOAD = 64 * 1024 * 1024


def _receive_exact(connection: socket.socket, count: int) -> bytes:
    chunks: list[bytes] = []
    received = 0
    while received < count:
        chunk = connection.recv(count - received)
        if not chunk:
            raise ConnectionError("局域网连接在接收完整数据前关闭")
        chunks.append(chunk)
        received += len(chunk)
    return b"".join(chunks)


class FrameCodec:
    def __init__(self, secret: bytes, max_payload: int = DEFAULT_MAX_PAYLOAD) -> None:
        if len(secret) < 32:
            raise ValueError("局域网密钥至少需要32字节")
        if max_payload <= 0:
            raise ValueError("最大负载必须大于零")
        self.secret = bytes(secret)
        self.max_payload = int(max_payload)

    def encode(self, channel: Channel, payload: bytes) -> bytes:
        raw = bytes(payload)
        if len(raw) > self.max_payload:
            raise ProtocolError("负载超过允许大小")
        header = HEADER.pack(MAGIC, int(channel), len(raw))
        signature = hmac.new(self.secret, header + raw, hashlib.sha256).digest()
        return header + raw + signature

    def decode(self, frame: bytes, expected_channel: Channel | None = None) -> tuple[Channel, bytes]:
        if len(frame) < HEADER.size + SIGNATURE_SIZE:
            raise ProtocolError("传递帧不完整")
        magic, channel_number, payload_size = HEADER.unpack(frame[: HEADER.size])
        if magic != MAGIC or payload_size > self.max_payload:
            raise ProtocolError("传递帧头无效")
        expected_size = HEADER.size + payload_size + SIGNATURE_SIZE
        if len(frame) != expected_size:
            raise ProtocolError("传递帧长度不一致")
        try:
            channel = Channel(channel_number)
        except ValueError as exc:
            raise ProtocolError("未知局域网通道") from exc
        if expected_channel is not None and channel != expected_channel:
            raise ProtocolError("数据到达了错误通道")
        payload_end = HEADER.size + payload_size
        payload = frame[HEADER.size:payload_end]
        supplied = frame[payload_end:]
        expected = hmac.new(self.secret, frame[:payload_end], hashlib.sha256).digest()
        if not hmac.compare_digest(supplied, expected):
            raise ProtocolError("传递帧校验失败")
        return channel, payload

    def send(self, connection: socket.socket, channel: Channel, payload: bytes) -> None:
        connection.sendall(self.encode(channel, payload))

    def receive(self, connection: socket.socket, expected_channel: Channel) -> bytes:
        header = _receive_exact(connection, HEADER.size)
        magic, channel_number, payload_size = HEADER.unpack(header)
        if magic != MAGIC or payload_size > self.max_payload:
            raise ProtocolError("传递帧头无效")
        remaining = _receive_exact(connection, payload_size + SIGNATURE_SIZE)
        _, payload = self.decode(header + remaining, expected_channel)
        if channel_number != int(expected_channel):
            raise ProtocolError("数据到达了错误通道")
        return payload

