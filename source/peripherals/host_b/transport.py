"""乙电脑到甲电脑的独立局域网连接。"""

from __future__ import annotations

from contextlib import AbstractContextManager
import socket
from threading import RLock
from types import TracebackType

from .config import NetworkConfig
from .entry_points import CHANNEL_CONFIG_NAMES, Channel
from .protocol import FrameCodec


class ChannelConnection(AbstractContextManager["ChannelConnection"]):
    """一条信息流使用一个独立连接，避免互相等待。"""

    def __init__(
        self,
        network: NetworkConfig,
        channel: Channel,
        *,
        connect_timeout: float = 5.0,
    ) -> None:
        self.network = network
        self.channel = channel
        self.connect_timeout = float(connect_timeout)
        self.codec = FrameCodec(network.secret)
        self.socket: socket.socket | None = None
        self._send_lock = RLock()
        self._receive_lock = RLock()

    @property
    def port(self) -> int:
        return self.network.ports[CHANNEL_CONFIG_NAMES[self.channel]]

    def connect(self) -> None:
        if self.socket is not None:
            return
        self.socket = socket.create_connection(
            (self.network.host_a_address, self.port),
            timeout=self.connect_timeout,
        )
        self.socket.settimeout(None)

    def send(self, payload: bytes) -> None:
        if self.socket is None:
            raise ConnectionError("局域网通道尚未连接")
        with self._send_lock:
            self.codec.send(self.socket, self.channel, payload)

    def receive(self) -> bytes:
        if self.socket is None:
            raise ConnectionError("局域网通道尚未连接")
        with self._receive_lock:
            return self.codec.receive(self.socket, self.channel)

    def close(self) -> None:
        if self.socket is not None:
            try:
                self.socket.shutdown(socket.SHUT_RDWR)
            except OSError:
                pass
            self.socket.close()
            self.socket = None

    def __enter__(self) -> "ChannelConnection":
        self.connect()
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        self.close()
