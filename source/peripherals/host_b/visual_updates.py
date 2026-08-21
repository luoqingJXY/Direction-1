"""视觉感受场的无损变化传递参考实现。

比较结果只用于减少局域网字节，不形成新的视觉活动。正式的1280×657
版本以后可以替换成图形处理器实现，但必须通过本文件的重建测试。
"""

from __future__ import annotations

from dataclasses import dataclass
import struct

from .errors import ProtocolError


UPDATE_HEADER = struct.Struct("!BHHI")
SPAN_HEADER = struct.Struct("!IH")
FULL_STATE = 0
CHANGED_SPANS = 1
RGB_CHANNELS = 3
MAX_SPAN_PIXELS = 0xFFFF


@dataclass(frozen=True, slots=True)
class VisualFrame:
    width: int
    height: int
    rgb: bytes

    def __post_init__(self) -> None:
        if self.width <= 0 or self.height <= 0:
            raise ValueError("视觉尺寸必须大于零")
        if len(self.rgb) != self.width * self.height * RGB_CHANNELS:
            raise ValueError("RGB数据量与视觉尺寸不一致")


@dataclass(frozen=True, slots=True)
class ChangedSpan:
    start_pixel: int
    rgb: bytes

    def __post_init__(self) -> None:
        if self.start_pixel < 0:
            raise ValueError("变化位置不能小于零")
        if not self.rgb or len(self.rgb) % RGB_CHANNELS:
            raise ValueError("一个变化段必须包含完整RGB位置")
        if self.pixel_count > MAX_SPAN_PIXELS:
            raise ValueError("一个变化段过长")

    @property
    def pixel_count(self) -> int:
        return len(self.rgb) // RGB_CHANNELS


@dataclass(frozen=True, slots=True)
class VisualUpdate:
    width: int
    height: int
    full_rgb: bytes | None = None
    spans: tuple[ChangedSpan, ...] = ()

    def __post_init__(self) -> None:
        if self.width <= 0 or self.height <= 0:
            raise ValueError("视觉尺寸必须大于零")
        if self.full_rgb is not None:
            if self.spans:
                raise ValueError("完整状态不能同时携带变化段")
            if len(self.full_rgb) != self.width * self.height * RGB_CHANNELS:
                raise ValueError("完整RGB状态长度错误")

    @property
    def is_full(self) -> bool:
        return self.full_rgb is not None

    @property
    def changed_pixel_count(self) -> int:
        if self.full_rgb is not None:
            return self.width * self.height
        return sum(span.pixel_count for span in self.spans)

    def encode(self) -> bytes:
        if self.full_rgb is not None:
            return UPDATE_HEADER.pack(FULL_STATE, self.width, self.height, 0) + self.full_rgb
        chunks = [UPDATE_HEADER.pack(CHANGED_SPANS, self.width, self.height, len(self.spans))]
        for span in self.spans:
            chunks.append(SPAN_HEADER.pack(span.start_pixel, span.pixel_count))
            chunks.append(span.rgb)
        return b"".join(chunks)

    @classmethod
    def decode(cls, payload: bytes) -> "VisualUpdate":
        if len(payload) < UPDATE_HEADER.size:
            raise ProtocolError("视觉变化数据不完整")
        kind, width, height, span_count = UPDATE_HEADER.unpack(payload[: UPDATE_HEADER.size])
        position = UPDATE_HEADER.size
        if kind == FULL_STATE:
            if span_count != 0:
                raise ProtocolError("完整视觉状态包含无效变化段数量")
            rgb = payload[position:]
            try:
                return cls(width, height, full_rgb=rgb)
            except ValueError as exc:
                raise ProtocolError(str(exc)) from exc
        if kind != CHANGED_SPANS:
            raise ProtocolError("未知视觉变化类型")
        spans: list[ChangedSpan] = []
        total_pixels = width * height
        for _ in range(span_count):
            if position + SPAN_HEADER.size > len(payload):
                raise ProtocolError("视觉变化段头不完整")
            start, count = SPAN_HEADER.unpack(payload[position : position + SPAN_HEADER.size])
            position += SPAN_HEADER.size
            byte_count = count * RGB_CHANNELS
            if count == 0 or start + count > total_pixels or position + byte_count > len(payload):
                raise ProtocolError("视觉变化段范围无效")
            spans.append(ChangedSpan(start, payload[position : position + byte_count]))
            position += byte_count
        if position != len(payload):
            raise ProtocolError("视觉变化数据尾部存在多余字节")
        return cls(width, height, spans=tuple(spans))


class ReferenceVisualChangeEncoder:
    """逐RGB位置精确比较；用于先确定传递语义和测试。"""

    def build(self, previous: VisualFrame | None, current: VisualFrame) -> VisualUpdate:
        if previous is None:
            return VisualUpdate(current.width, current.height, full_rgb=current.rgb)
        if (previous.width, previous.height) != (current.width, current.height):
            raise ValueError("视觉尺寸变化后必须重新发送完整状态")

        spans: list[ChangedSpan] = []
        total_pixels = current.width * current.height
        pixel = 0
        while pixel < total_pixels:
            offset = pixel * RGB_CHANNELS
            if previous.rgb[offset : offset + RGB_CHANNELS] == current.rgb[offset : offset + RGB_CHANNELS]:
                pixel += 1
                continue
            start = pixel
            changed = bytearray()
            while pixel < total_pixels and pixel - start < MAX_SPAN_PIXELS:
                offset = pixel * RGB_CHANNELS
                old_rgb = previous.rgb[offset : offset + RGB_CHANNELS]
                new_rgb = current.rgb[offset : offset + RGB_CHANNELS]
                if old_rgb == new_rgb:
                    break
                changed.extend(new_rgb)
                pixel += 1
            spans.append(ChangedSpan(start, bytes(changed)))
        return VisualUpdate(current.width, current.height, spans=tuple(spans))


def apply_visual_update(previous: VisualFrame | None, update: VisualUpdate) -> VisualFrame:
    if update.full_rgb is not None:
        return VisualFrame(update.width, update.height, update.full_rgb)
    if previous is None:
        raise ValueError("首次视觉更新必须是完整状态")
    if (previous.width, previous.height) != (update.width, update.height):
        raise ValueError("视觉变化尺寸与现有受体状态不一致")
    rgb = bytearray(previous.rgb)
    total_pixels = update.width * update.height
    for span in update.spans:
        if span.start_pixel + span.pixel_count > total_pixels:
            raise ValueError("视觉变化段超出受体范围")
        start = span.start_pixel * RGB_CHANNELS
        rgb[start : start + len(span.rgb)] = span.rgb
    return VisualFrame(update.width, update.height, bytes(rgb))
