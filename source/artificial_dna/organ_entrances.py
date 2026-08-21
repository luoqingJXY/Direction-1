"""全部已确认器官活动在同一神经组织中的紧凑出生入口。"""

from __future__ import annotations

from dataclasses import dataclass

import numpy

from .birth_structure import TissueAddress, VisualReceptorAddress
from .brain_geometry import TISSUE_PLANE


VISUAL_ACTIVITY_COUNT = 1280 * 657 * 3
AUDITORY_ACTIVITY_COUNT = 3 * 1025 * 3
PREDICTED_VISUAL_ACTIVITY_COUNT = 512 * 512
PREDICTED_AUDITORY_ACTIVITY_COUNT = 3 * 342 * 85 * 3
MOUSE_ACTIVITY_COUNT = 4
KEYBOARD_ACTIVITY_COUNT = 108
VIEW_CENTER_ACTIVITY_COUNT = 4
ORGAN_VOLUME_DEPTH = 18


@dataclass(frozen=True, slots=True)
class EntranceRange:
    name: str
    activity_count: int
    activity_offset: int


@dataclass(frozen=True, slots=True)
class EntrancePair:
    receiver: TissueAddress
    ordinary: TissueAddress


class OrganEntranceLayout:
    """用地址计算表达六百多万入口神经元，不生成数百万个对象。"""

    def __init__(self) -> None:
        counts = (
            ("visual", VISUAL_ACTIVITY_COUNT),
            ("auditory", AUDITORY_ACTIVITY_COUNT),
            ("predicted_visual", PREDICTED_VISUAL_ACTIVITY_COUNT),
            ("predicted_auditory", PREDICTED_AUDITORY_ACTIVITY_COUNT),
            ("mouse", MOUSE_ACTIVITY_COUNT),
            ("keyboard", KEYBOARD_ACTIVITY_COUNT),
            ("view_center", VIEW_CENTER_ACTIVITY_COUNT),
        )
        ranges: list[EntranceRange] = []
        offset = 0
        for name, count in counts:
            ranges.append(EntranceRange(name, count, offset))
            offset += count
        self.ranges = tuple(ranges)
        self._by_name = {value.name: value for value in self.ranges}
        self.activity_count = offset
        self.neuron_count = 2 * offset
        # 入口必须保留各器官自身的二维/多维邻接，并让固定接收端与普通端
        # 位于相邻深度面；所以物理占址末端不是简单的神经元数量相加。
        self.next_free_index = ORGAN_VOLUME_DEPTH * TISSUE_PLANE

    @staticmethod
    def _paired_address(x: int, y: int, fixed_z: int, ordinary: bool) -> TissueAddress:
        return TissueAddress(x, y, fixed_z + int(ordinary))

    @classmethod
    def _physical_pair(cls, entrance: str, index: int) -> EntrancePair:
        if entrance == "visual":
            pixel, channel = divmod(index, 3)
            y, x = divmod(pixel, 1280)
            tile, local_x = divmod(x, 256)
            cell_x = local_x * 3 + channel
            fixed_z = tile * 2
            return EntrancePair(
                cls._paired_address(cell_x, y, fixed_z, False),
                cls._paired_address(cell_x, y, fixed_z, True),
            )

        if entrance == "auditory":
            stream, remainder = divmod(index, 1025 * 3)
            frequency, component = divmod(remainder, 3)
            tile, local_frequency = divmod(frequency, 256)
            x = local_frequency * 3 + component
            y = stream * 6 + tile
            return EntrancePair(
                cls._paired_address(x, y, 10, False),
                cls._paired_address(x, y, 10, True),
            )

        if entrance == "predicted_visual":
            y, x = divmod(index, 512)
            return EntrancePair(
                cls._paired_address(x, y, 12, False),
                cls._paired_address(x, y, 12, True),
            )

        if entrance == "predicted_auditory":
            stream, remainder = divmod(index, 342 * 85 * 3)
            frequency, remainder = divmod(remainder, 85 * 3)
            sequence, component = divmod(remainder, 3)
            tile, local_frequency = divmod(frequency, 256)
            x = local_frequency * 3 + component
            y = (stream * 2 + tile) * 85 + sequence
            return EntrancePair(
                cls._paired_address(x, y, 14, False),
                cls._paired_address(x, y, 14, True),
            )

        if entrance == "mouse":
            x, y = index, 0
        elif entrance == "keyboard":
            x, y = index, 1
        elif entrance == "view_center":
            x, y = index, 2
        else:  # pragma: no cover - pair已先验证名称
            raise ValueError("未知器官入口")
        return EntrancePair(
            cls._paired_address(x, y, 16, False),
            cls._paired_address(x, y, 16, True),
        )

    def pair(self, entrance: str, activity_index: int) -> EntrancePair:
        try:
            group = self._by_name[entrance]
        except KeyError as exc:
            raise ValueError("未知器官入口") from exc
        index = int(activity_index)
        if not 0 <= index < group.activity_count:
            raise ValueError("器官活动编号超出入口范围")
        return self._physical_pair(entrance, index)

    def visual_pair(self, receptor: VisualReceptorAddress) -> EntrancePair:
        index = (receptor.y * 1280 + receptor.x) * 3 + int(receptor.channel)
        return self.pair("visual", index)

    def receiver_indices(
        self,
        entrance: str,
        activity_indices: numpy.ndarray,
    ) -> numpy.ndarray:
        """把一组器官活动编号直接落到各自固定接收神经元地址。

        这是入口物理排列的批量版本。数组中的每一项仍对应一个活动和一个
        固定接收神经元；这里只避免在一次真实画面到达时创建数百万个 Python
        对象，不会合并、平均或解释活动。
        """

        try:
            group = self._by_name[entrance]
        except KeyError as exc:
            raise ValueError("未知器官入口") from exc
        indices = numpy.asarray(activity_indices)
        if indices.ndim != 1 or not numpy.issubdtype(indices.dtype, numpy.integer):
            raise ValueError("器官活动编号必须是一维整数排列")
        if indices.size and (
            numpy.any(indices < 0) or numpy.any(indices >= group.activity_count)
        ):
            raise ValueError("器官活动编号超出入口范围")
        values = indices.astype(numpy.int64, copy=False)

        if entrance == "visual":
            pixel = values // 3
            channel = values % 3
            y = pixel // 1280
            x = pixel % 1280
            tile = x // 256
            result_x = (x % 256) * 3 + channel
            result_y = y
            result_z = tile * 2
        elif entrance == "auditory":
            stream = values // (1025 * 3)
            remainder = values % (1025 * 3)
            frequency = remainder // 3
            component = remainder % 3
            tile = frequency // 256
            result_x = (frequency % 256) * 3 + component
            result_y = stream * 6 + tile
            result_z = numpy.full(values.shape, 10, dtype=numpy.int64)
        elif entrance == "predicted_visual":
            result_y = values // 512
            result_x = values % 512
            result_z = numpy.full(values.shape, 12, dtype=numpy.int64)
        elif entrance == "predicted_auditory":
            stream = values // (342 * 85 * 3)
            remainder = values % (342 * 85 * 3)
            frequency = remainder // (85 * 3)
            remainder %= 85 * 3
            sequence = remainder // 3
            component = remainder % 3
            tile = frequency // 256
            result_x = (frequency % 256) * 3 + component
            result_y = (stream * 2 + tile) * 85 + sequence
            result_z = numpy.full(values.shape, 14, dtype=numpy.int64)
        elif entrance == "mouse":
            result_x = values
            result_y = numpy.zeros(values.shape, dtype=numpy.int64)
            result_z = numpy.full(values.shape, 16, dtype=numpy.int64)
        elif entrance == "keyboard":
            result_x = values
            result_y = numpy.ones(values.shape, dtype=numpy.int64)
            result_z = numpy.full(values.shape, 16, dtype=numpy.int64)
        elif entrance == "view_center":
            result_x = values
            result_y = numpy.full(values.shape, 2, dtype=numpy.int64)
            result_z = numpy.full(values.shape, 16, dtype=numpy.int64)
        else:  # pragma: no cover - 上方group已经限定入口名称
            raise AssertionError("器官入口分派不完整")

        return (
            result_z * TISSUE_PLANE + result_y * 800 + result_x
        ).astype(numpy.uint32, copy=False)

    def ordinary_indices(
        self,
        entrance: str,
        activity_indices: numpy.ndarray,
    ) -> numpy.ndarray:
        """给出一组器官活动各自唯一普通入口的物理地址。"""

        receivers = self.receiver_indices(entrance, activity_indices)
        return (
            receivers + numpy.uint32(TISSUE_PLANE)
        ).astype(numpy.uint32, copy=False)


SECOND_EXPERIMENT_ENTRANCES = OrganEntranceLayout()
