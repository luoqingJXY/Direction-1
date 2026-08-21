"""乙电脑声音设备与真实动作执行的一次性本机配置。"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from pathlib import Path
import json
import shutil
import tempfile

from .audio_capture import AudioDevice, list_audio_devices
from .actuation_rules import SECOND_EXPERIMENT_KEYBOARD_SCAN_CODES
from .config import load_config


def _choose_device(
    candidates: Sequence[AudioDevice],
    description: str,
    *,
    input_text: Callable[[str], str],
    output_text: Callable[[str], None],
) -> AudioDevice:
    if not candidates:
        raise RuntimeError(f"没有发现可用于{description}的48,000次采集设备")
    output_text(f"\n可用于{description}的设备：")
    for number, device in enumerate(candidates, 1):
        output_text(
            f"  {number}. {device.name} "
            f"（输入{device.input_channels}声道，{device.sample_rate}次/秒）"
        )
    if len(candidates) == 1:
        selected = candidates[0]
        output_text(f"已自动选择：{selected.name}")
        return selected
    while True:
        answer = input_text(f"请输入{description}设备前的数字：").strip()
        try:
            number = int(answer)
        except ValueError:
            number = 0
        if 1 <= number <= len(candidates):
            return candidates[number - 1]
        output_text("输入无效，请重新输入。")


def _replace_table_values(
    text: str,
    replacements: dict[tuple[str, str], str],
) -> str:
    current_table = ""
    replaced: set[tuple[str, str]] = set()
    lines = text.splitlines(keepends=True)
    result: list[str] = []
    for line in lines:
        stripped = line.strip()
        if stripped.startswith("[") and stripped.endswith("]"):
            current_table = stripped[1:-1].strip()
        changed = False
        for (table, key), value in replacements.items():
            if current_table == table and stripped.startswith(f"{key} ="):
                ending = "\r\n" if line.endswith("\r\n") else "\n" if line.endswith("\n") else ""
                result.append(f"{key} = {value}{ending}")
                replaced.add((table, key))
                changed = True
                break
        if not changed:
            result.append(line)
    missing = set(replacements) - replaced
    if missing:
        names = "、".join(f"[{table}] {key}" for table, key in sorted(missing))
        raise RuntimeError(f"配置文件缺少需要修改的项目：{names}")
    return "".join(result)


def configure_peripherals(
    config_path: str | Path,
    *,
    devices: Sequence[AudioDevice] | None = None,
    input_text: Callable[[str], str] = input,
    output_text: Callable[[str], None] = print,
) -> Path:
    """选择两类真实设备并在明确确认后解锁受限键鼠执行。"""

    path = Path(config_path)
    load_config(path)
    available = tuple(list_audio_devices() if devices is None else devices)
    computer_candidates = tuple(
        device
        for device in available
        if device.loopback
        and device.input_channels >= 2
        and device.sample_rate == 48_000
    )
    microphone_candidates = tuple(
        device
        for device in available
        if not device.loopback
        and device.input_channels >= 1
        and device.sample_rate == 48_000
    )
    computer = _choose_device(
        computer_candidates,
        "电脑声音",
        input_text=input_text,
        output_text=output_text,
    )
    microphone = _choose_device(
        microphone_candidates,
        "麦克风",
        input_text=input_text,
        output_text=output_text,
    )

    output_text("\n键鼠执行端只允许Minecraft处于前台时工作。")
    output_text("Win、Alt、Esc、F11、F12、电源键和系统菜单键始终被锁定。")
    answer = input_text("确认让模型实际操作其余受限键鼠时，请输入 ARM：").strip()
    if answer != "ARM":
        raise RuntimeError("没有收到ARM确认，配置未改变，键鼠仍保持锁定")

    original = path.read_text(encoding="utf-8")
    changed = _replace_table_values(
        original,
        {
            ("visual", "nonlinear_mapping_confirmed"): "true",
            ("visual", "low_frequency_operation_confirmed"): "true",
            ("audio", "computer_output_device"): json.dumps(
                computer.name,
                ensure_ascii=False,
            ),
            ("audio", "microphone_device"): json.dumps(
                microphone.name,
                ensure_ascii=False,
            ),
            ("audio", "auditory_field_confirmed"): "true",
            ("actuation", "armed"): "true",
            ("actuation", "mouse_rule_confirmed"): "true",
            ("actuation", "keyboard_mapping_confirmed"): "true",
            ("actuation", "view_center_rule_confirmed"): "true",
            ("actuation", "keyboard_scan_codes"): json.dumps(
                list(SECOND_EXPERIMENT_KEYBOARD_SCAN_CODES)
            ),
        },
    )
    backup = path.with_name(path.name + ".before_peripheral_setup")
    if not backup.exists():
        shutil.copy2(path, backup)
    temporary_name = ""
    try:
        with tempfile.NamedTemporaryFile(
            "w",
            encoding="utf-8",
            newline="",
            dir=path.parent,
            prefix=path.name + ".",
            suffix=".pending",
            delete=False,
        ) as temporary:
            temporary.write(changed)
            temporary_name = temporary.name
        pending = Path(temporary_name)
        configured = load_config(pending)
        errors = configured.readiness_errors()
        if errors:
            raise RuntimeError("完整外围配置仍未闭合：" + "；".join(errors))
        pending.replace(path)
    finally:
        if temporary_name:
            Path(temporary_name).unlink(missing_ok=True)

    output_text(f"\n完整外围已经配置：{path}")
    output_text(f"原配置备份：{backup}")
    return path
