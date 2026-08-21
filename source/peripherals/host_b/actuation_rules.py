"""已确认活动与真实设备动作之间的规则切入点。"""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Protocol

from second_experiment.common.activities import KeyboardActivities, MouseActivities

from .actuation import FORBIDDEN_SCAN_CODES, ResolvedAction
from .visual_processing import VisualCenterState


class MouseActuationRule(Protocol):
    def resolve(self, activities: MouseActivities) -> ResolvedAction: ...


class KeyboardActuationRule(Protocol):
    def resolve(self, activities: KeyboardActivities) -> ResolvedAction: ...


class VisualCenterActuationRule(Protocol):
    def apply(self, activities: MouseActivities, state: VisualCenterState) -> None: ...


@dataclass(frozen=True, slots=True)
class ConfirmedVisualCenterActuationRule:
    """把四路连续活动直接形成真实视野中心位移。

    一次收到的完整四路活动就是一次真实发生。强度一对应原始画面的一个
    位置，不设置动作阈值，也不引入时间、目标、自动注意或位置输入信号。
    两个相反方向同时到达时，其实际位移直接相减；只有画面物理边界会截断
    最终位置。
    """

    raw_width: int = 3840
    raw_height: int = 2160

    def __post_init__(self) -> None:
        if self.raw_width <= 0 or self.raw_height <= 0:
            raise ValueError("原始视觉尺寸必须大于零")

    def apply(self, activities: MouseActivities, state: VisualCenterState) -> None:
        values = activities.values
        if any(value > 1.0 for value in values):
            raise ValueError("视野中心四路活动必须处于[0,1]")
        horizontal = state.horizontal + (
            activities.x_positive - activities.x_negative
        ) / self.raw_width
        vertical = state.vertical + (
            activities.y_positive - activities.y_negative
        ) / self.raw_height
        state.set_position(
            min(1.0, max(0.0, horizontal)),
            min(1.0, max(0.0, vertical)),
        )


class ConfirmedMouseActuationRule:
    """四路连续活动直接形成Windows相对鼠标运动。"""

    def __init__(self) -> None:
        self._horizontal_remainder = 0.0
        self._vertical_remainder = 0.0

    def resolve(self, activities: MouseActivities) -> ResolvedAction:
        if any(value > 1.0 for value in activities.values):
            raise ValueError("鼠标四路活动必须处于[0,1]")
        horizontal = self._horizontal_remainder + (
            activities.x_positive - activities.x_negative
        )
        vertical = self._vertical_remainder + (
            activities.y_positive - activities.y_negative
        )
        dx = math.trunc(horizontal)
        dy = math.trunc(vertical)
        self._horizontal_remainder = horizontal - dx
        self._vertical_remainder = vertical - dy
        return ResolvedAction(mouse_dx=dx, mouse_dy=dy)


# 108项控制入口的顺序属于客观键盘器官的出生接触关系，不进入Brain信号。
# 前88项按Windows第一套扫描码1..88排列，随后是20项E0扩展键；锁定项用0
# 保留其控制入口位置，但永远不形成实际系统按键。
_KEYBOARD_EXTENDED_SCAN_CODES = (
    0xE01C,
    0xE01D,
    0xE035,
    0xE037,
    0xE038,
    0xE045,
    0xE047,
    0xE048,
    0xE049,
    0xE04B,
    0xE04D,
    0xE04F,
    0xE050,
    0xE051,
    0xE052,
    0xE053,
    0xE05B,
    0xE05C,
    0xE05D,
    0xE05E,
)
SECOND_EXPERIMENT_KEYBOARD_SCAN_CODES = tuple(
    0 if value in FORBIDDEN_SCAN_CODES else value
    for value in (*range(1, 0x59), *_KEYBOARD_EXTENDED_SCAN_CODES)
)


@dataclass(frozen=True, slots=True)
class ConfirmedKeyboardActuationRule:
    """108项连续活动各自控制一个已配置且安全的实际键。"""

    scan_codes: tuple[int, ...]
    threshold: float = 0.5

    def __post_init__(self) -> None:
        codes = tuple(int(value) for value in self.scan_codes)
        object.__setattr__(self, "scan_codes", codes)
        if len(codes) != 108:
            raise ValueError("键盘必须恰好配置108项扫描码")
        active_codes = tuple(value for value in codes if value != 0)
        if len(set(active_codes)) != len(active_codes):
            raise ValueError("两个键盘活动不能控制同一个实际按键")
        if any(
            value < 0
            or (
                value > 0xFF
                and not ((value & 0xFF00) == 0xE000 and (value & 0xFF) != 0)
            )
            for value in codes
        ):
            raise ValueError("键盘扫描码超出支持范围")
        if any(value in FORBIDDEN_SCAN_CODES for value in active_codes):
            raise ValueError("配置包含已经锁定的系统或中止按键")
        value = float(self.threshold)
        if not math.isfinite(value) or not 0.0 <= value <= 1.0:
            raise ValueError("键盘按下阈值必须处于[0,1]")

    def resolve(self, activities: KeyboardActivities) -> ResolvedAction:
        if any(value > 1.0 for value in activities.values):
            raise ValueError("108项键盘活动必须处于[0,1]")
        return ResolvedAction(
            key_states=tuple(
                (scan_code, activity >= self.threshold)
                for scan_code, activity in zip(self.scan_codes, activities.values)
                if scan_code != 0
            )
        )


if len(SECOND_EXPERIMENT_KEYBOARD_SCAN_CODES) != 108:
    raise RuntimeError("第二次实验键盘出生接触关系不再是108项")
