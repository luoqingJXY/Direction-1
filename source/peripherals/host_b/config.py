"""乙电脑的集中配置。

配置中的 ``*_confirmed`` 不是生命信号，也不会进入脑。它们只保证
尚未闭合的规则不会被临时代码替代后误用于正式实验。
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import tomllib
from typing import Any

from .errors import ConfigurationError


REQUIRED_CHANNELS = (
    "visual",
    "audio",
    "mouse",
    "keyboard",
    "view_center",
    "voice",
    "control",
)


def _table(data: dict[str, Any], name: str) -> dict[str, Any]:
    value = data.get(name)
    if not isinstance(value, dict):
        raise ConfigurationError(f"缺少配置段：{name}")
    return value


def _text(data: dict[str, Any], name: str, *, allow_empty: bool = False) -> str:
    value = data.get(name)
    if not isinstance(value, str) or (not allow_empty and not value.strip()):
        raise ConfigurationError(f"配置项 {name} 必须是字符串")
    return value.strip()


def _integer(data: dict[str, Any], name: str, *, minimum: int = 0) -> int:
    value = data.get(name)
    if not isinstance(value, int) or isinstance(value, bool) or value < minimum:
        raise ConfigurationError(f"配置项 {name} 必须是不小于 {minimum} 的整数")
    return value


def _boolean(data: dict[str, Any], name: str) -> bool:
    value = data.get(name)
    if not isinstance(value, bool):
        raise ConfigurationError(f"配置项 {name} 必须是 true 或 false")
    return value


@dataclass(frozen=True, slots=True)
class NetworkConfig:
    host_a_address: str
    secret_hex: str
    ports: dict[str, int]

    @property
    def secret(self) -> bytes:
        try:
            secret = bytes.fromhex(self.secret_hex)
        except ValueError as exc:
            raise ConfigurationError("network.secret_hex 不是有效的十六进制字符串") from exc
        if len(secret) < 32:
            raise ConfigurationError("network.secret_hex 至少需要 32 字节")
        return secret


@dataclass(frozen=True, slots=True)
class MinecraftConfig:
    title_contains: str
    process_names: tuple[str, ...]
    require_foreground_for_actions: bool


@dataclass(frozen=True, slots=True)
class VisualConfig:
    width: int
    height: int
    capture_backend: str
    nonlinear_mapping_confirmed: bool
    low_frequency_operation_confirmed: bool


@dataclass(frozen=True, slots=True)
class AudioConfig:
    computer_output_device: str
    microphone_device: str
    auditory_field_confirmed: bool


@dataclass(frozen=True, slots=True)
class ActuationConfig:
    armed: bool
    mouse_rule_confirmed: bool
    keyboard_mapping_confirmed: bool
    view_center_rule_confirmed: bool
    keyboard_scan_codes: tuple[int, ...]


@dataclass(frozen=True, slots=True)
class HostBConfig:
    network: NetworkConfig
    minecraft: MinecraftConfig
    visual: VisualConfig
    audio: AudioConfig
    actuation: ActuationConfig

    def readiness_errors(self) -> tuple[str, ...]:
        """返回阻止正式生命运行的项目，不检查普通诊断能力。"""

        errors: list[str] = []
        if (self.visual.width, self.visual.height) != (1280, 657):
            errors.append("第二次实验正式视觉输入必须保持 1280×657")
        if not self.visual.nonlinear_mapping_confirmed:
            errors.append("视野中心非线性空间变换的具体函数尚未确认")
        if not self.visual.low_frequency_operation_confirmed:
            errors.append("去低频操作的具体规则尚未确认")
        if not self.audio.auditory_field_confirmed:
            errors.append("听觉感受场具体排列尚未确认")
        if self.audio.auditory_field_confirmed and not self.audio.computer_output_device:
            errors.append("尚未指定电脑声音的48,000次采集设备")
        if self.audio.auditory_field_confirmed and not self.audio.microphone_device:
            errors.append("尚未指定麦克风的48,000次采集设备")
        if not self.actuation.mouse_rule_confirmed:
            errors.append("四路鼠标活动形成实际位移的规则尚未确认")
        if not self.actuation.keyboard_mapping_confirmed:
            errors.append("108个键盘活动与实际按键的对应尚未确认")
        if not self.actuation.view_center_rule_confirmed:
            errors.append("视野中心运动活动的方向规则尚未确认")
        if not self.actuation.armed:
            errors.append("鼠标和键盘实际执行端尚未确认解锁")
        if self.actuation.keyboard_mapping_confirmed and len(self.actuation.keyboard_scan_codes) != 108:
            errors.append("确认键盘映射后必须恰好配置108个扫描码")
        return tuple(errors)


def load_config(path: str | Path) -> HostBConfig:
    config_path = Path(path)
    try:
        with config_path.open("rb") as stream:
            data = tomllib.load(stream)
    except OSError as exc:
        raise ConfigurationError(f"无法读取配置文件：{config_path}") from exc
    except tomllib.TOMLDecodeError as exc:
        raise ConfigurationError(f"配置文件格式错误：{exc}") from exc

    network_data = _table(data, "network")
    ports_data = _table(network_data, "ports")
    ports = {name: _integer(ports_data, name, minimum=1) for name in REQUIRED_CHANNELS}
    if len(set(ports.values())) != len(ports):
        raise ConfigurationError("每条局域网通道必须使用不同端口")
    network = NetworkConfig(
        host_a_address=_text(network_data, "host_a_address"),
        secret_hex=_text(network_data, "secret_hex"),
        ports=ports,
    )
    network.secret

    minecraft_data = _table(data, "minecraft")
    process_names = minecraft_data.get("process_names")
    if not isinstance(process_names, list) or not process_names or not all(
        isinstance(item, str) and item.strip() for item in process_names
    ):
        raise ConfigurationError("minecraft.process_names 必须是非空字符串数组")
    minecraft = MinecraftConfig(
        title_contains=_text(minecraft_data, "title_contains"),
        process_names=tuple(item.strip() for item in process_names),
        require_foreground_for_actions=_boolean(minecraft_data, "require_foreground_for_actions"),
    )

    visual_data = _table(data, "visual")
    visual = VisualConfig(
        width=_integer(visual_data, "width", minimum=1),
        height=_integer(visual_data, "height", minimum=1),
        capture_backend=_text(visual_data, "capture_backend"),
        nonlinear_mapping_confirmed=_boolean(visual_data, "nonlinear_mapping_confirmed"),
        low_frequency_operation_confirmed=_boolean(visual_data, "low_frequency_operation_confirmed"),
    )

    audio_data = _table(data, "audio")
    audio = AudioConfig(
        computer_output_device=_text(audio_data, "computer_output_device", allow_empty=True),
        microphone_device=_text(audio_data, "microphone_device", allow_empty=True),
        auditory_field_confirmed=_boolean(audio_data, "auditory_field_confirmed"),
    )

    actuation_data = _table(data, "actuation")
    scan_codes = actuation_data.get("keyboard_scan_codes")
    if not isinstance(scan_codes, list) or not all(
        isinstance(item, int) and not isinstance(item, bool) and 0 <= item <= 0xFFFF
        for item in scan_codes
    ):
        raise ConfigurationError("actuation.keyboard_scan_codes 必须是扫描码整数数组")
    actuation = ActuationConfig(
        armed=_boolean(actuation_data, "armed"),
        mouse_rule_confirmed=_boolean(actuation_data, "mouse_rule_confirmed"),
        keyboard_mapping_confirmed=_boolean(actuation_data, "keyboard_mapping_confirmed"),
        view_center_rule_confirmed=_boolean(actuation_data, "view_center_rule_confirmed"),
        keyboard_scan_codes=tuple(scan_codes),
    )

    return HostBConfig(network, minecraft, visual, audio, actuation)
