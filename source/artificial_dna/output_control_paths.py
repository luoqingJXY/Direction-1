"""已确定形状的脑输出到客观器官控制入口的固定出生路径。

统一公式规定 ``Y^Brain = O_D(Q+)``，因此输出器官必须接收到 Path 当前
传播活动，不能直接把某些 Neuron 当前活动当作器官输出。现有分节结构末端
与五组客观器官控制入口逐项一对一接触；这段边界固定路径只保留末端活动，
不负责产生动作幅度或再次削弱，因此人工DNA中的出生强度固定为1。
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import IntEnum
from typing import Iterator

import numpy

from .brain_dna_layout import (
    motion_output_address,
    predicted_auditory_output_address,
    visual_reconstruction_address,
)
from .brain_geometry import linear_index
from .organ_entrances import (
    KEYBOARD_ACTIVITY_COUNT,
    MOUSE_ACTIVITY_COUNT,
    PREDICTED_AUDITORY_ACTIVITY_COUNT,
    PREDICTED_VISUAL_ACTIVITY_COUNT,
    VIEW_CENTER_ACTIVITY_COUNT,
)


class KnownOutputGroup(IntEnum):
    """工程核对编号；不作为标签进入 Brain 或器官活动。"""

    PREDICTED_VISUAL = 0
    PREDICTED_AUDITORY = 1
    MOUSE = 2
    KEYBOARD = 3
    VIEW_CENTER = 4


KNOWN_OUTPUT_GROUP_COUNTS = (
    PREDICTED_VISUAL_ACTIVITY_COUNT,
    PREDICTED_AUDITORY_ACTIVITY_COUNT,
    MOUSE_ACTIVITY_COUNT,
    KEYBOARD_ACTIVITY_COUNT,
    VIEW_CENTER_ACTIVITY_COUNT,
)
KNOWN_OUTPUT_GROUP_OFFSETS = tuple(
    sum(KNOWN_OUTPUT_GROUP_COUNTS[:index])
    for index in range(len(KNOWN_OUTPUT_GROUP_COUNTS))
)
KNOWN_OUTPUT_CONTROL_PATH_COUNT = sum(KNOWN_OUTPUT_GROUP_COUNTS)
KNOWN_OUTPUT_CONTROL_PATH_BIRTH_STRENGTH = 1.0


OUTPUT_CONTROL_PATH_DTYPE = numpy.dtype(
    [
        ("source_neuron", "<u4"),
        ("control_entrance", "<u4"),
        ("path_strength", "<f4"),
        ("current_activity", "<f4"),
    ]
)

OUTPUT_CONTROL_PATH_GENE_DTYPE = numpy.dtype(
    [
        ("source_neuron", "<u4"),
        ("control_entrance", "<u4"),
        ("path_strength", "<f4"),
    ]
)


@dataclass(frozen=True, slots=True)
class OutputControlPathEndpoint:
    source_neuron: int
    control_entrance: int


def _index(address) -> int:
    return linear_index(address.x, address.y, address.z)


def iter_known_output_control_path_endpoints() -> Iterator[OutputControlPathEndpoint]:
    """按器官入口顺序给出五组已知输出的来源和控制入口。"""

    control = 0
    for y in range(512):
        for x in range(512):
            yield OutputControlPathEndpoint(
                _index(visual_reconstruction_address(0, x, y)),
                control,
            )
            control += 1

    for stream in range(3):
        for frequency in range(342):
            for sequence in range(85):
                for component in range(3):
                    yield OutputControlPathEndpoint(
                        _index(
                            predicted_auditory_output_address(
                                stream,
                                frequency,
                                sequence,
                                component,
                            )
                        ),
                        control,
                    )
                    control += 1

    for action in range(
        MOUSE_ACTIVITY_COUNT + KEYBOARD_ACTIVITY_COUNT + VIEW_CENTER_ACTIVITY_COUNT
    ):
        yield OutputControlPathEndpoint(
            _index(motion_output_address(action)),
            control,
        )
        control += 1

    if control != KNOWN_OUTPUT_CONTROL_PATH_COUNT:  # pragma: no cover
        raise RuntimeError("已知输出器官固定路径数量与器官入口总数不一致")


def build_known_output_control_path_genes_by_source() -> numpy.ndarray:
    """形成523890条正式出生记录，并按来源地址排列供分块运行。"""

    values = numpy.empty(
        KNOWN_OUTPUT_CONTROL_PATH_COUNT,
        dtype=OUTPUT_CONTROL_PATH_GENE_DTYPE,
    )
    produced = 0
    for endpoint in iter_known_output_control_path_endpoints():
        values[produced] = (
            endpoint.source_neuron,
            endpoint.control_entrance,
            KNOWN_OUTPUT_CONTROL_PATH_BIRTH_STRENGTH,
        )
        produced += 1
    if produced != KNOWN_OUTPUT_CONTROL_PATH_COUNT:  # pragma: no cover
        raise RuntimeError("输出器官固定路径出生记录数量不正确")
    values.sort(order=("source_neuron", "control_entrance"), kind="stable")
    controls = numpy.sort(values["control_entrance"])
    if not numpy.array_equal(
        controls,
        numpy.arange(KNOWN_OUTPUT_CONTROL_PATH_COUNT, dtype="<u4"),
    ):
        raise RuntimeError("输出器官控制入口没有保持逐项唯一接触")
    if numpy.any(values["source_neuron"][1:] < values["source_neuron"][:-1]):
        raise RuntimeError("输出器官固定路径出生记录没有按来源地址排列")
    if not numpy.all(
        values["path_strength"] == KNOWN_OUTPUT_CONTROL_PATH_BIRTH_STRENGTH
    ):
        raise RuntimeError("输出器官边界固定路径没有保持原活动")
    return values


def validate_known_output_control_path_endpoints() -> None:
    count = 0
    previous_control = -1
    for endpoint in iter_known_output_control_path_endpoints():
        if endpoint.control_entrance != previous_control + 1:
            raise RuntimeError("已知输出器官控制入口没有保持逐项一对一顺序")
        previous_control = endpoint.control_entrance
        count += 1
    if count != KNOWN_OUTPUT_CONTROL_PATH_COUNT:
        raise RuntimeError("已知输出器官固定路径总数不正确")


validate_known_output_control_path_endpoints()
