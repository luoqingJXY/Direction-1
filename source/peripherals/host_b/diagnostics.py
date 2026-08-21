"""乙电脑本机诊断；结果只供控制台观察。"""

from __future__ import annotations

from dataclasses import asdict
import hashlib
import json
from pathlib import Path
import platform
import sys
from typing import Any

from .audio_capture import Float32AudioCapture, list_audio_devices
from second_experiment.common.rgb_reference import write_rgb_ppm
from .config import HostBConfig
from .minecraft_window import MinecraftWindowLocator, select_minecraft_window
from .visual_capture import MssMinecraftCapture
from .visual_processing import VisualCenterState, build_confirmed_visual_processing


def platform_report() -> dict[str, Any]:
    return {
        "system": platform.system(),
        "release": platform.release(),
        "machine": platform.machine(),
        "python": platform.python_version(),
        "is_64_bit": sys.maxsize > 2**32,
    }


def readiness_report(config: HostBConfig) -> dict[str, Any]:
    errors = list(config.readiness_errors())
    audio_probe = None
    if not errors:
        captures: list[Float32AudioCapture] = []
        try:
            computer = Float32AudioCapture(
                config.audio.computer_output_device,
                channels=2,
                require_loopback=True,
            )
            captures.append(computer)
            microphone = Float32AudioCapture(
                config.audio.microphone_device,
                channels=1,
                require_loopback=False,
            )
            captures.append(microphone)
            computer_bytes = computer.read()
            microphone_bytes = microphone.read()
            expected_computer = computer.frames_per_block * computer.channels * 4
            expected_microphone = microphone.frames_per_block * microphone.channels * 4
            if len(computer_bytes) != expected_computer:
                raise RuntimeError("电脑声音没有返回完整的192个双声道采样")
            if len(microphone_bytes) != expected_microphone:
                raise RuntimeError("麦克风没有返回完整的192个单声道采样")
            audio_probe = {
                "computer_output_samples": computer.frames_per_block,
                "computer_output_channels": computer.channels,
                "microphone_samples": microphone.frames_per_block,
                "microphone_channels": microphone.channels,
                "sample_rate": 48_000,
            }
        except (OSError, RuntimeError, ValueError) as exc:
            errors.append(f"声音设备实际打开和采样失败：{exc}")
        finally:
            for capture in reversed(captures):
                capture.close()
    return {
        "ready_for_formal_life_run": not errors,
        "unconfirmed_items": errors,
        "visual_size": [config.visual.width, config.visual.height],
        "host_a_address": config.network.host_a_address,
        "actuation_armed": config.actuation.armed,
        "audio_probe": audio_probe,
    }


def minecraft_window_report(config: HostBConfig) -> dict[str, Any]:
    locator = MinecraftWindowLocator(config.minecraft)
    candidates = locator.candidates()
    window = select_minecraft_window(
        candidates,
        locator.title_contains,
        locator.process_names,
    )
    relevant = [
        candidate
        for candidate in candidates
        if (
            locator.title_contains in candidate.title.casefold()
            or candidate.process_name.casefold() in locator.process_names
            or candidate.width * candidate.height >= 640 * 360
        )
    ]
    relevant.sort(key=lambda item: -(item.width * item.height))
    return {
        "found": window is not None,
        "window": asdict(window) if window else None,
        "configured_title": config.minecraft.title_contains,
        "configured_processes": list(config.minecraft.process_names),
        "visible_candidates": [asdict(candidate) for candidate in relevant],
    }


def audio_device_report() -> dict[str, Any]:
    return {"devices": [asdict(device) for device in list_audio_devices()]}


def capture_one_frame_report(config: HostBConfig) -> dict[str, Any]:
    locator = MinecraftWindowLocator(config.minecraft)
    capture = MssMinecraftCapture(locator)
    try:
        frame, window = capture.read()
    finally:
        capture.close()
    return {
        "width": frame.width,
        "height": frame.height,
        "rgb_bytes": len(frame.rgb),
        "rgb_sha256": hashlib.sha256(frame.rgb).hexdigest(),
        "window_title": window.title,
        "foreground": window.foreground,
    }


def capture_visual_reference(
    config: HostBConfig,
    output_directory: Path,
    name: str,
    *,
    center_horizontal: float = 0.5,
    center_vertical: float = 0.5,
) -> dict[str, Any]:
    """保存人工DNA取址所需的原始画面和最终视觉活动画面。"""

    if not config.visual.nonlinear_mapping_confirmed:
        raise RuntimeError("视野中心空间变换尚未确认，不能形成正式视觉参考")
    if not config.visual.low_frequency_operation_confirmed:
        raise RuntimeError("去低频操作尚未确认，不能形成正式视觉参考")
    reference_name = name.strip()
    allowed = "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789-_"
    if not reference_name or any(character not in allowed for character in reference_name):
        raise ValueError("参考画面名称只能包含字母、数字、减号和下划线")
    center = VisualCenterState()
    center.set_position(center_horizontal, center_vertical)
    locator = MinecraftWindowLocator(config.minecraft)
    capture = MssMinecraftCapture(locator)
    try:
        raw, window = capture.read()
    finally:
        capture.close()
    processed = build_confirmed_visual_processing(
        config.visual.width,
        config.visual.height,
    ).process(raw, center)

    directory = Path(output_directory)
    directory.mkdir(parents=True, exist_ok=True)
    raw_path = write_rgb_ppm(raw, directory / f"{reference_name}.raw.ppm")
    processed_path = write_rgb_ppm(
        processed,
        directory / f"{reference_name}.processed.ppm",
    )
    report = {
        "name": reference_name,
        "width": processed.width,
        "height": processed.height,
        "visual_center": [center.horizontal, center.vertical],
        "raw_rgb_sha256": hashlib.sha256(raw.rgb).hexdigest(),
        "processed_rgb_sha256": hashlib.sha256(processed.rgb).hexdigest(),
        "raw_file": raw_path.name,
        "processed_file": processed_path.name,
        "window_title": window.title,
        "foreground": window.foreground,
    }
    report_path = directory / f"{reference_name}.json"
    report_path.write_text(
        json.dumps(report, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return report
