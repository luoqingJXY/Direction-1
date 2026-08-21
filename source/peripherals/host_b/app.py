"""乙电脑源码阶段的命令入口。"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import secrets

from .config import load_config
from .diagnostics import (
    audio_device_report,
    capture_one_frame_report,
    capture_visual_reference,
    minecraft_window_report,
    platform_report,
    readiness_report,
)
from .actuation import WindowsInputSink, emergency_key_pressed
from .actuation_rules import (
    ConfirmedKeyboardActuationRule,
    ConfirmedMouseActuationRule,
    ConfirmedVisualCenterActuationRule,
)
from .auditory_flow import build_confirmed_continuous_auditory_flow
from .body import ArtificialBody
from .flow_service import HostBFlowService
from .runtime import HostBRuntime
from .minecraft_window import MinecraftWindowLocator
from .peripheral_setup import configure_peripherals
from .visual_flow import build_confirmed_continuous_visual_flow
from .visual_processing import VisualCenterState
from .teaching_input import WindowsTeachingInputCapture


def _print_json(value: object) -> None:
    print(json.dumps(value, ensure_ascii=False, indent=2))


def _initialize_config(output: Path) -> None:
    if output.exists():
        raise FileExistsError(f"配置文件已经存在：{output}")
    example = Path(__file__).with_name("host_b.example.toml")
    text = example.read_text(encoding="utf-8")
    text = text.replace('secret_hex = "请替换"', f'secret_hex = "{secrets.token_hex(32)}"')
    output.write_text(text, encoding="utf-8")
    print(f"已生成可编辑配置：{output}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="第二次实验乙电脑源码工具")
    subcommands = parser.add_subparsers(dest="command", required=True)

    initialize = subcommands.add_parser(
        "initialize",
        aliases=["初始化配置"],
        help="生成一份带随机局域网密钥的配置",
    )
    initialize.add_argument("输出", type=Path)

    check = subcommands.add_parser("check", aliases=["检查"], help="检查配置和当前未确认项目")
    check.add_argument("配置", type=Path)

    window = subcommands.add_parser(
        "check-window",
        aliases=["检查窗口"],
        help="检查Minecraft窗口定位",
    )
    window.add_argument("配置", type=Path)

    subcommands.add_parser("audio-devices", aliases=["声音设备"], help="列出Windows声音设备")

    setup = subcommands.add_parser(
        "setup-peripherals",
        aliases=["配置完整外围"],
        help="选择声音设备并明确解锁受限键鼠执行",
    )
    setup.add_argument("配置", type=Path)

    capture = subcommands.add_parser(
        "capture-one",
        aliases=["捕捉一帧"],
        help="捕捉一份未变换的1280×657 RGB诊断画面",
    )
    capture.add_argument("配置", type=Path)

    reference = subcommands.add_parser(
        "capture-reference",
        aliases=["保存视觉参考"],
        help="保存原始画面和最终1280×657视觉活动，供人工DNA固定路径取址",
    )
    reference.add_argument("配置", type=Path)
    reference.add_argument("输出目录", type=Path)
    reference.add_argument("名称")
    reference.add_argument("--center-x", type=float, default=0.5)
    reference.add_argument("--center-y", type=float, default=0.5)

    subcommands.add_parser("system", aliases=["系统"], help="显示乙电脑基础环境")
    run_visual = subcommands.add_parser(
        "run-visual-link",
        aliases=["运行视觉连接"],
        help="连接甲电脑的七条通道并持续发送已确认视觉活动",
    )
    run_visual.add_argument("配置", type=Path)
    run_life = subcommands.add_parser(
        "run-life-link",
        aliases=["运行完整外围"],
        help="检查全部真实外围后连接视觉、听觉、动作和工程控制",
    )
    run_life.add_argument("配置", type=Path)
    return parser


def _run_peripheral_link(config, *, require_formal_readiness: bool) -> None:
    if require_formal_readiness:
        errors = config.readiness_errors()
        if errors:
            raise RuntimeError("正式外围尚不能启动：" + "；".join(errors))
    locator = MinecraftWindowLocator(config.minecraft)
    if require_formal_readiness and locator.locate() is None:
        raise RuntimeError("没有找到唯一的Minecraft窗口，完整外围不能启动")
    center = VisualCenterState()
    body = ArtificialBody(
        WindowsInputSink(
            locator,
            config.minecraft.require_foreground_for_actions,
        ),
        center,
    )
    runtime: HostBRuntime
    teaching_input = WindowsTeachingInputCapture(
        teaching_active=lambda: body.teaching_active,
        capture_allowed=lambda: (
            (window := locator.locate()) is not None and window.foreground
        ),
        send_mouse=lambda activities: runtime.send_teacher_mouse(activities),
        send_keyboard=lambda activities: runtime.send_teacher_keyboard(activities),
    )

    def start_teaching() -> None:
        body.start_teaching()
        teaching_input.begin_teaching()

    def stop_teaching() -> None:
        teaching_input.finish_teaching()
        body.stop_teaching()

    runtime = HostBRuntime(
        config,
        start_teaching=start_teaching,
        stop_teaching=stop_teaching,
        close_components=body.close,
    )
    visual = build_confirmed_continuous_visual_flow(
        config,
        center,
        runtime.send_visual_update,
    )
    auditory = None
    if (
        config.audio.auditory_field_confirmed
        and config.audio.computer_output_device
        and config.audio.microphone_device
    ):
        auditory = build_confirmed_continuous_auditory_flow(
            config,
            runtime.send_auditory_update,
        )
    service = HostBFlowService(
        runtime,
        visual,
        auditory,
        visual_center_rule=(
            ConfirmedVisualCenterActuationRule()
            if config.actuation.view_center_rule_confirmed
            else None
        ),
        mouse_rule=(
            ConfirmedMouseActuationRule()
            if config.actuation.armed and config.actuation.mouse_rule_confirmed
            else None
        ),
        keyboard_rule=(
            ConfirmedKeyboardActuationRule(config.actuation.keyboard_scan_codes)
            if config.actuation.armed and config.actuation.keyboard_mapping_confirmed
            else None
        ),
        apply_input_action=body.apply_mouse_and_keyboard,
        teaching_input=teaching_input,
        external_stop=emergency_key_pressed,
    )
    try:
        service.run()
    except KeyboardInterrupt:
        service.close()


def main(argv: list[str] | None = None) -> int:
    arguments = build_parser().parse_args(argv)
    if arguments.command in {"initialize", "初始化配置"}:
        _initialize_config(arguments.输出)
        return 0
    if arguments.command in {"system", "系统"}:
        _print_json(platform_report())
        return 0
    if arguments.command in {"audio-devices", "声音设备"}:
        _print_json(audio_device_report())
        return 0

    config = load_config(arguments.配置)
    if arguments.command in {"setup-peripherals", "配置完整外围"}:
        configure_peripherals(arguments.配置)
        return 0
    if arguments.command in {"run-life-link", "运行完整外围"}:
        _run_peripheral_link(config, require_formal_readiness=True)
        return 0
    if arguments.command in {"run-visual-link", "运行视觉连接"}:
        _run_peripheral_link(config, require_formal_readiness=False)
        return 0
    if arguments.command in {"check", "检查"}:
        _print_json(readiness_report(config))
        return 0
    if arguments.command in {"check-window", "检查窗口"}:
        _print_json(minecraft_window_report(config))
        return 0
    if arguments.command in {"capture-one", "捕捉一帧"}:
        _print_json(capture_one_frame_report(config))
        return 0
    if arguments.command in {"capture-reference", "保存视觉参考"}:
        _print_json(
            capture_visual_reference(
                config,
                arguments.输出目录,
                arguments.名称,
                center_horizontal=arguments.center_x,
                center_vertical=arguments.center_y,
            )
        )
        return 0
    raise AssertionError("未处理的命令")


if __name__ == "__main__":
    raise SystemExit(main())
