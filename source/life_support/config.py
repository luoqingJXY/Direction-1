"""甲电脑集中配置。"""

from __future__ import annotations

from dataclasses import dataclass
from ipaddress import IPv4Address
from pathlib import Path
import tomllib
from typing import Any

from second_experiment.host_b.config import REQUIRED_CHANNELS, load_config as load_host_b_config
from second_experiment.host_b.errors import ConfigurationError


@dataclass(frozen=True, slots=True)
class HostANetworkConfig:
    bind_address: str
    allowed_peer_address: str
    secret_hex: str
    ports: dict[str, int]

    @property
    def secret(self) -> bytes:
        try:
            result = bytes.fromhex(self.secret_hex)
        except ValueError as exc:
            raise ConfigurationError("network.secret_hex 不是有效十六进制字符串") from exc
        if len(result) < 32:
            raise ConfigurationError("network.secret_hex 至少需要32字节")
        return result


@dataclass(frozen=True, slots=True)
class HostAConfig:
    network: HostANetworkConfig
    life_storage_directory: Path
    observation_file: Path


def _ipv4(value: Any, name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ConfigurationError(f"{name} 必须是IPv4地址")
    try:
        return str(IPv4Address(value.strip()))
    except ValueError as exc:
        raise ConfigurationError(f"{name} 必须是IPv4地址") from exc


def _table(data: dict[str, Any], name: str) -> dict[str, Any]:
    value = data.get(name)
    if not isinstance(value, dict):
        raise ConfigurationError(f"缺少配置段：{name}")
    return value


def load_config(path: str | Path) -> HostAConfig:
    source = Path(path)
    try:
        with source.open("rb") as stream:
            data = tomllib.load(stream)
    except (OSError, tomllib.TOMLDecodeError) as exc:
        raise ConfigurationError(f"无法读取甲电脑配置：{source}") from exc

    network_data = _table(data, "network")
    ports_data = _table(network_data, "ports")
    bind_address = _ipv4(network_data.get("bind_address"), "network.bind_address")
    allowed_peer_address = _ipv4(
        network_data.get("allowed_peer_address"),
        "network.allowed_peer_address",
    )
    secret_hex = network_data.get("secret_hex")
    if not isinstance(secret_hex, str):
        raise ConfigurationError("network.secret_hex 必须是字符串")
    ports: dict[str, int] = {}
    for name in REQUIRED_CHANNELS:
        port = ports_data.get(name)
        if not isinstance(port, int) or isinstance(port, bool) or not 1 <= port <= 65535:
            raise ConfigurationError(f"network.ports.{name} 必须是有效端口")
        ports[name] = port
    if len(set(ports.values())) != len(ports):
        raise ConfigurationError("每条局域网通道必须使用不同端口")
    network = HostANetworkConfig(
        bind_address,
        allowed_peer_address,
        secret_hex.strip(),
        ports,
    )
    network.secret

    storage_data = _table(data, "storage")
    life_directory = storage_data.get("life_directory")
    observation_file = storage_data.get("observation_file")
    if not isinstance(life_directory, str) or not life_directory.strip():
        raise ConfigurationError("storage.life_directory 必须是非空路径")
    if not isinstance(observation_file, str) or not observation_file.strip():
        raise ConfigurationError("storage.observation_file 必须是非空路径")
    base = source.parent
    life_path = Path(life_directory)
    observation_path = Path(observation_file)
    if not life_path.is_absolute():
        life_path = base / life_path
    if not observation_path.is_absolute():
        observation_path = base / observation_path
    return HostAConfig(network, life_path, observation_path)


def create_config_from_host_b(
    host_b_path: str | Path,
    output_path: str | Path,
    *,
    bind_address: str,
    allowed_peer_address: str,
) -> Path:
    """从乙电脑配置形成严格匹配的甲电脑私有配置。"""

    source = load_host_b_config(host_b_path)
    bind = _ipv4(bind_address, "bind_address")
    peer = _ipv4(allowed_peer_address, "allowed_peer_address")
    destination = Path(output_path)
    if destination.exists():
        raise FileExistsError(f"配置文件已经存在：{destination}")

    port_lines = "\n".join(
        f"{name} = {source.network.ports[name]}" for name in REQUIRED_CHANNELS
    )
    contents = (
        "[network]\n"
        f'bind_address = "{bind}"\n'
        f'allowed_peer_address = "{peer}"\n'
        f'secret_hex = "{source.network.secret_hex}"\n\n'
        "[network.ports]\n"
        f"{port_lines}\n\n"
        "[storage]\n"
        'life_directory = "life_storage"\n'
        'observation_file = "observations.jsonl"\n'
    )
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(contents, encoding="utf-8")
    load_config(destination)
    return destination
