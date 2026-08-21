"""视觉中心空间变换和去低频操作的可替换切入点。"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

from .visual_updates import VisualFrame


@dataclass(slots=True)
class VisualCenterState:
    """视野中心运动器官在乙电脑中造成的实际中心状态。"""

    horizontal: float = 0.5
    vertical: float = 0.5

    def set_position(self, horizontal: float, vertical: float) -> None:
        x = float(horizontal)
        y = float(vertical)
        if not 0.0 <= x <= 1.0 or not 0.0 <= y <= 1.0:
            raise ValueError("视野中心实际位置必须处于画面范围")
        self.horizontal = x
        self.vertical = y


SpatialTransform = Callable[[VisualFrame, VisualCenterState], VisualFrame]
LowFrequencyOperation = Callable[[VisualFrame], VisualFrame]

LOW_FREQUENCY_ZERO_RADIUS = 1.0 / 256.0
LOW_FREQUENCY_FULL_RADIUS = 1.0 / 128.0


def _numpy_module():
    try:
        import numpy
    except ImportError as exc:
        raise RuntimeError("正式视觉处理需要安装 numpy") from exc
    return numpy


def _source_indices(
    source_length: int,
    output_length: int,
    center: float,
    numpy,
) -> object:
    if output_length == 1 or source_length == 1:
        return numpy.zeros(output_length, dtype=numpy.intp)

    positions = numpy.linspace(0.0, 1.0, output_length, dtype=numpy.float64)
    source = numpy.empty(output_length, dtype=numpy.float64)
    left = positions < center
    right = ~left

    if center > 0.0:
        distance = (center - positions[left]) / center
        source[left] = center - center * distance**2
    else:
        source[left] = 0.0

    if center < 1.0:
        distance = (positions[right] - center) / (1.0 - center)
        source[right] = center + (1.0 - center) * distance**2
    else:
        source[right] = 1.0

    source = numpy.clip(source, 0.0, 1.0)
    return numpy.rint(source * (source_length - 1)).astype(numpy.intp)


def confirmed_spatial_transform(
    raw_frame: VisualFrame,
    center: VisualCenterState,
    output_width: int | None = None,
    output_height: int | None = None,
) -> VisualFrame:
    """按照冻结的平方关系重新分配原始RGB位置。"""

    numpy = _numpy_module()
    width = raw_frame.width if output_width is None else int(output_width)
    height = raw_frame.height if output_height is None else int(output_height)
    if width <= 0 or height <= 0:
        raise ValueError("视觉感受场尺寸必须大于零")
    rgb = numpy.frombuffer(raw_frame.rgb, dtype=numpy.uint8).reshape(
        raw_frame.height,
        raw_frame.width,
        3,
    )
    source_x = _source_indices(raw_frame.width, width, center.horizontal, numpy)
    source_y = _source_indices(raw_frame.height, height, center.vertical, numpy)
    transformed = numpy.ascontiguousarray(rgb[source_y[:, None], source_x[None, :], :])
    return VisualFrame(width, height, transformed.tobytes())


def _fixed_frequency_mask(height: int, width: int, numpy) -> object:
    horizontal = numpy.abs(numpy.fft.fftfreq(width))
    vertical = numpy.abs(numpy.fft.fftfreq(height))
    radius = numpy.sqrt(vertical[:, None] ** 2 + horizontal[None, :] ** 2)
    mask = numpy.ones((height, width), dtype=numpy.float64)
    mask[radius <= LOW_FREQUENCY_ZERO_RADIUS] = 0.0
    transition = (radius > LOW_FREQUENCY_ZERO_RADIUS) & (
        radius < LOW_FREQUENCY_FULL_RADIUS
    )
    mask[transition] = (
        radius[transition] - LOW_FREQUENCY_ZERO_RADIUS
    ) / (LOW_FREQUENCY_FULL_RADIUS - LOW_FREQUENCY_ZERO_RADIUS)
    return mask


def confirmed_low_frequency_operation(frame: VisualFrame) -> VisualFrame:
    """分别处理三个RGB平面，并以固定中点恢复非负活动。"""

    numpy = _numpy_module()
    rgb = numpy.frombuffer(frame.rgb, dtype=numpy.uint8).reshape(
        frame.height,
        frame.width,
        3,
    )
    values = rgb.astype(numpy.float64) / 255.0
    frequencies = numpy.fft.fft2(values, axes=(0, 1))
    mask = _fixed_frequency_mask(frame.height, frame.width, numpy)
    restored = numpy.fft.ifft2(frequencies * mask[:, :, None], axes=(0, 1)).real
    nonnegative = numpy.clip(0.5 + restored / 2.0, 0.0, 1.0)
    encoded = numpy.rint(nonnegative * 255.0).astype(numpy.uint8)
    return VisualFrame(frame.width, frame.height, encoded.tobytes())


def build_confirmed_visual_processing(
    output_width: int = 1280,
    output_height: int = 657,
) -> "ConfirmedVisualProcessing":
    return ConfirmedVisualProcessing(
        lambda frame, center: confirmed_spatial_transform(
            frame,
            center,
            output_width,
            output_height,
        ),
        confirmed_low_frequency_operation,
        output_width=output_width,
        output_height=output_height,
    )


class ConfirmedVisualProcessing:
    """依次执行已确认的视野中心空间变换和去低频操作。"""

    def __init__(
        self,
        spatial_transform: SpatialTransform,
        low_frequency_operation: LowFrequencyOperation,
        *,
        output_width: int | None = None,
        output_height: int | None = None,
    ) -> None:
        if (output_width is None) != (output_height is None):
            raise ValueError("视觉感受场宽度和高度必须同时指定")
        if output_width is not None and (output_width <= 0 or output_height <= 0):
            raise ValueError("视觉感受场尺寸必须大于零")
        self.spatial_transform = spatial_transform
        self.low_frequency_operation = low_frequency_operation
        self.output_width = output_width
        self.output_height = output_height

    def process(self, raw_frame: VisualFrame, center: VisualCenterState) -> VisualFrame:
        transformed = self.spatial_transform(raw_frame, center)
        expected = (
            (raw_frame.width, raw_frame.height)
            if self.output_width is None
            else (self.output_width, self.output_height)
        )
        if (transformed.width, transformed.height) != expected:
            raise ValueError("视野中心空间变换没有形成指定视觉感受场尺寸")
        processed = self.low_frequency_operation(transformed)
        if (processed.width, processed.height) != expected:
            raise ValueError("去低频操作不能改变视觉分辨率")
        return processed
