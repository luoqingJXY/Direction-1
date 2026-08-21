"""已确认固定路径强度在完整固定路径总账中的逐条出生记录。

固定路径端点总账中仍含尚待确认的分节草案，所以不能为全部记录填入默认
强度。本文件只展开已经由人工出生结构确定的路径，并保存其完整总账编号
和强度。

总账编号和批次类别只用于人工DNA写入，不进入Brain的Signal。
"""

from __future__ import annotations

from dataclasses import dataclass
from itertools import islice
from typing import Iterator

import numpy

from .fixed_path_topology import (
    CONFIRMED_FIXED_PATH_STRENGTH_COUNT,
    CONFIRMED_FIXED_PATH_STRENGTH_FAMILIES,
    FIXED_PATH_FAMILY_COUNTS,
    SECOND_EXPERIMENT_ACTIVE_FIXED_PATH_COUNT,
    SECOND_EXPERIMENT_ACTIVE_FIXED_PATH_FAMILIES,
    FixedPathFamily,
    FixedPathTopology,
)


FIXED_PATH_BIRTH_STRENGTH_DTYPE = numpy.dtype(
    [("path_index", "<u4"), ("strength", "<f4")]
)


def _family_prefixes() -> dict[FixedPathFamily, int]:
    prefixes: dict[FixedPathFamily, int] = {}
    current = 0
    for family in FixedPathFamily:
        prefixes[family] = current
        current += FIXED_PATH_FAMILY_COUNTS[family]
    return prefixes


FIXED_PATH_FAMILY_PREFIXES = _family_prefixes()


@dataclass(frozen=True, slots=True)
class ConfirmedFixedPathBirthStrengthBatch:
    family: FixedPathFamily
    values: numpy.ndarray

    def __post_init__(self) -> None:
        if self.values.dtype != FIXED_PATH_BIRTH_STRENGTH_DTYPE:
            raise ValueError("固定路径出生强度记录格式不正确")
        if self.values.ndim != 1:
            raise ValueError("固定路径出生强度记录必须是一维排列")

    @property
    def count(self) -> int:
        return int(self.values.size)


def _batch(
    family: FixedPathFamily,
    indexes: numpy.ndarray,
    strengths: numpy.ndarray,
) -> ConfirmedFixedPathBirthStrengthBatch:
    if indexes.size != strengths.size:
        raise ValueError("固定路径编号与出生强度数量不一致")
    values = numpy.empty(indexes.size, dtype=FIXED_PATH_BIRTH_STRENGTH_DTYPE)
    values["path_index"] = indexes
    values["strength"] = strengths
    return ConfirmedFixedPathBirthStrengthBatch(family, values)


def iter_confirmed_family_birth_strength_batches(
    topology: FixedPathTopology,
    family: FixedPathFamily,
    *,
    batch_size: int = 262_144,
) -> Iterator[ConfirmedFixedPathBirthStrengthBatch]:
    """展开一个已确认类别；未确认类别不以任何默认值代替。"""

    size = int(batch_size)
    if size <= 0:
        raise ValueError("固定路径出生强度批次大小必须大于零")
    selected = FixedPathFamily(family)
    prefix = FIXED_PATH_FAMILY_PREFIXES[selected]

    if selected in CONFIRMED_FIXED_PATH_STRENGTH_FAMILIES:
        strengths = topology.iter_confirmed_strengths(selected)
        produced = 0
        while True:
            chunk = tuple(islice(strengths, size))
            if not chunk:
                break
            count = len(chunk)
            indexes = numpy.arange(
                prefix + produced,
                prefix + produced + count,
                dtype="<u4",
            )
            yield _batch(
                selected,
                indexes,
                numpy.asarray(chunk, dtype="<f4"),
            )
            produced += count
        expected = FIXED_PATH_FAMILY_COUNTS[selected]
        if produced != expected:
            raise RuntimeError("已确认固定路径强度没有覆盖完整连接类别")
        return

    raise ValueError("该固定连接类别的路径强度尚未由人工出生结构确定")


def iter_all_confirmed_fixed_path_birth_strength_batches(
    topology: FixedPathTopology,
    *,
    batch_size: int = 262_144,
) -> Iterator[ConfirmedFixedPathBirthStrengthBatch]:
    """按完整路径总账顺序给出全部已确认固定路径出生强度。"""

    produced = 0
    for family in FixedPathFamily:
        if family not in CONFIRMED_FIXED_PATH_STRENGTH_FAMILIES:
            continue
        for batch in iter_confirmed_family_birth_strength_batches(
            topology,
            family,
            batch_size=batch_size,
        ):
            yield batch
            produced += batch.count
    if produced != CONFIRMED_FIXED_PATH_STRENGTH_COUNT:
        raise RuntimeError("固定路径出生强度记录与已确认总账数量不一致")


def iter_second_experiment_fixed_path_birth_strength_batches(
    topology: FixedPathTopology,
    *,
    batch_size: int = 262_144,
) -> Iterator[ConfirmedFixedPathBirthStrengthBatch]:
    """按本次实验紧凑端点顺序给出强度，不保留未出生草案的编号空洞。"""

    size = int(batch_size)
    if size <= 0:
        raise ValueError("固定路径出生强度批次大小必须大于零")
    path_index = 0
    for family in SECOND_EXPERIMENT_ACTIVE_FIXED_PATH_FAMILIES:
        strengths = topology.iter_confirmed_strengths(family)
        family_count = 0
        while True:
            chunk = tuple(islice(strengths, size))
            if not chunk:
                break
            count = len(chunk)
            indexes = numpy.arange(
                path_index,
                path_index + count,
                dtype="<u4",
            )
            yield _batch(
                family,
                indexes,
                numpy.asarray(chunk, dtype="<f4"),
            )
            family_count += count
            path_index += count
        if family_count != FIXED_PATH_FAMILY_COUNTS[family]:
            raise RuntimeError("第二次实验固定路径强度没有覆盖完整连接类别")
    if path_index != SECOND_EXPERIMENT_ACTIVE_FIXED_PATH_COUNT:
        raise RuntimeError("第二次实验固定路径强度与端点出生范围不一致")


def validate_confirmed_fixed_path_birth_strength_layout() -> None:
    total = sum(FIXED_PATH_FAMILY_COUNTS.values())
    last = FixedPathFamily(max(value.value for value in FixedPathFamily))
    if FIXED_PATH_FAMILY_PREFIXES[last] + FIXED_PATH_FAMILY_COUNTS[last] != total:
        raise RuntimeError("固定路径类别前缀没有覆盖完整端点总账")
    if total > numpy.iinfo(numpy.uint32).max:
        raise RuntimeError("固定路径总账编号超过32位无符号整数范围")


validate_confirmed_fixed_path_birth_strength_layout()
