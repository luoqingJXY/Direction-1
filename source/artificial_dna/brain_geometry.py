"""第二次实验完整神经组织的唯一几何与地址规则。"""

from __future__ import annotations

from dataclasses import dataclass
from itertools import product


TISSUE_WIDTH = 800
TISSUE_HEIGHT = 800
TISSUE_DEPTH = 160
TISSUE_PLANE = TISSUE_WIDTH * TISSUE_HEIGHT
TISSUE_NEURON_COUNT = TISSUE_PLANE * TISSUE_DEPTH

# 顺序属于硬盘文件格式的一部分；以后不能在不迁移个体存档的情况下改变。
DIRECT_NEIGHBOR_OFFSETS = tuple(
    (dx, dy, dz)
    for dz, dy, dx in product((-1, 0, 1), repeat=3)
    if (dx, dy, dz) != (0, 0, 0)
)

if len(DIRECT_NEIGHBOR_OFFSETS) != 26:  # pragma: no cover - 导入时完整性断言
    raise RuntimeError("直接相邻方向必须恰好为26个")

DIRECTED_LOCAL_PATH_COUNT = sum(
    (TISSUE_WIDTH - abs(dx))
    * (TISSUE_HEIGHT - abs(dy))
    * (TISSUE_DEPTH - abs(dz))
    for dx, dy, dz in DIRECT_NEIGHBOR_OFFSETS
)


def validate_coordinates(x: int, y: int, z: int) -> None:
    if not (
        0 <= int(x) < TISSUE_WIDTH
        and 0 <= int(y) < TISSUE_HEIGHT
        and 0 <= int(z) < TISSUE_DEPTH
    ):
        raise ValueError("神经组织地址超出800×800×160逻辑空间")


def linear_index(x: int, y: int, z: int) -> int:
    """把三维组织地址变成唯一线性地址。"""

    validate_coordinates(x, y, z)
    return (int(z) * TISSUE_HEIGHT + int(y)) * TISSUE_WIDTH + int(x)


def coordinates(index: int) -> tuple[int, int, int]:
    """把唯一线性地址还原为三维组织地址。"""

    value = int(index)
    if not 0 <= value < TISSUE_NEURON_COUNT:
        raise ValueError("神经元线性地址超出完整神经组织容量")
    z, remainder = divmod(value, TISSUE_PLANE)
    y, x = divmod(remainder, TISSUE_WIDTH)
    return x, y, z


def neighbor_index(index: int, direction: int) -> int | None:
    """返回某一直接相邻位置的线性地址；边界外返回空。"""

    x, y, z = coordinates(index)
    try:
        direction_index = int(direction)
    except (TypeError, ValueError) as exc:
        raise ValueError("相邻方向编号必须处于0到25") from exc
    if not 0 <= direction_index < len(DIRECT_NEIGHBOR_OFFSETS):
        raise ValueError("相邻方向编号必须处于0到25")
    dx, dy, dz = DIRECT_NEIGHBOR_OFFSETS[direction_index]
    target = (x + dx, y + dy, z + dz)
    if not (
        0 <= target[0] < TISSUE_WIDTH
        and 0 <= target[1] < TISSUE_HEIGHT
        and 0 <= target[2] < TISSUE_DEPTH
    ):
        return None
    return linear_index(*target)


@dataclass(frozen=True, slots=True)
class LinearAddressRange:
    """半开区间[start, stop)，用于保证出生结构只占用唯一地址。"""

    name: str
    start: int
    stop: int

    def __post_init__(self) -> None:
        if not self.name:
            raise ValueError("地址区间必须有名称")
        if not 0 <= int(self.start) <= int(self.stop) <= TISSUE_NEURON_COUNT:
            raise ValueError("地址区间超出完整神经组织容量")

    @property
    def count(self) -> int:
        return self.stop - self.start

    def contains(self, index: int) -> bool:
        return self.start <= int(index) < self.stop
