"""甲电脑为每种信息流建立独立监听连接。"""

from __future__ import annotations

import socket
from threading import RLock, Thread

from second_experiment.host_b.entry_points import CHANNEL_CONFIG_NAMES, Channel
from second_experiment.host_b.protocol import FrameCodec

from .config import HostANetworkConfig


class ChannelServer:
    def __init__(self, config: HostANetworkConfig, channel: Channel) -> None:
        self.config = config
        self.channel = channel
        self.codec = FrameCodec(config.secret)
        self.listener: socket.socket | None = None
        self.connection: socket.socket | None = None
        self._send_lock = RLock()
        self._receive_lock = RLock()

    @property
    def port(self) -> int:
        return self.config.ports[CHANNEL_CONFIG_NAMES[self.channel]]

    def listen(self) -> None:
        if self.listener is not None:
            return
        listener = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        listener.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        listener.bind((self.config.bind_address, self.port))
        listener.listen(1)
        self.listener = listener

    def accept(self) -> None:
        if self.listener is None:
            raise RuntimeError("通道尚未开始监听")
        if self.connection is not None:
            return
        while self.connection is None:
            connection, address = self.listener.accept()
            if address[0] != self.config.allowed_peer_address:
                connection.close()
                continue
            connection.settimeout(None)
            self.connection = connection

    def send(self, payload: bytes) -> None:
        if self.connection is None:
            raise ConnectionError("乙电脑尚未连接该通道")
        with self._send_lock:
            self.codec.send(self.connection, self.channel, payload)

    def receive(self) -> bytes:
        if self.connection is None:
            raise ConnectionError("乙电脑尚未连接该通道")
        with self._receive_lock:
            return self.codec.receive(self.connection, self.channel)

    def close(self) -> None:
        if self.connection is not None:
            try:
                self.connection.shutdown(socket.SHUT_RDWR)
            except OSError:
                pass
            self.connection.close()
            self.connection = None
        if self.listener is not None:
            self.listener.close()
            self.listener = None


class HostANetwork:
    def __init__(self, config: HostANetworkConfig) -> None:
        self.channels = {channel: ChannelServer(config, channel) for channel in Channel}

    def listen_all(self) -> None:
        try:
            for server in self.channels.values():
                server.listen()
        except BaseException:
            self.close_all()
            raise

    def accept(self, channel: Channel) -> None:
        self.channels[channel].accept()

    def accept_all(self) -> None:
        """七条监听已经建立后，并行等待乙电脑的七条连接。"""

        errors: list[BaseException] = []
        error_lock = RLock()

        def accept_one(server: ChannelServer) -> None:
            try:
                server.accept()
            except BaseException as exc:
                with error_lock:
                    errors.append(exc)

        workers = [
            Thread(target=accept_one, args=(server,), name=f"accept-{server.channel.name}")
            for server in self.channels.values()
        ]
        for worker in workers:
            worker.start()
        for worker in workers:
            worker.join()
        if errors:
            self.close_all()
            raise ConnectionError("未能建立全部七条乙电脑连接") from errors[0]

    @property
    def fully_connected(self) -> bool:
        return all(server.connection is not None for server in self.channels.values())

    def send(self, channel: Channel, payload: bytes) -> None:
        self.channels[channel].send(payload)

    def receive(self, channel: Channel) -> bytes:
        return self.channels[channel].receive()

    def close_all(self) -> None:
        for server in self.channels.values():
            server.close()
