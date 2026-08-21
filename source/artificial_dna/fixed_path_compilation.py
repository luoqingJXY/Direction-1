"""当前固定路径端点结构的硬盘编译。

固定路径拓扑按出生结构中的连接类别生成，便于核对完整性；实际运行却必须
按来源神经元取出当前核心组织的所有路径。本文件只把已经确定的来源—到达
端点整理为按来源地址的稳定硬盘排列。

它不创建人工生命状态，也不把未确认的路径强度、当前传播活动、阈值或调制
响应写成零值。已确认类别和仍待确认的结构草案也不能在清单中混称为正式
人工DNA；人工DNA完整后才可共同出生。
"""

from __future__ import annotations

import json
import tempfile
from pathlib import Path
from typing import Iterable

import numpy

from .brain_geometry import TISSUE_NEURON_COUNT
from .fixed_path_topology import (
    FIXED_PATH_ENDPOINT_DTYPE,
    FixedPathEndpointBatch,
    FixedPathTopology,
)


# 当前版本在完整RGB视觉接触的基础上，确认三段动作形成通道，
# 并让116项动作各自的5个独立末端位置到达对应输出神经元。
FORMAT_VERSION = 11
TOPOLOGY_LAYOUT_VERSION = "real-auditory-one-to-one-association-contact-v10"
MANIFEST_FILE_NAME = "fixed_path_structure_manifest.json"
ENDPOINT_FILE_NAME = "fixed_path_endpoints_by_source.bin"


def formal_source_partition_stops() -> tuple[int, ...]:
    """按正式运行的六个核心组织给出来源地址分区边界。

    分区只是一次性硬盘整理和之后分块装载共同使用的边界，不是 Region、
    Signal 或人工DNA的新结构。
    """

    from .brain_chunks import BrainChunkPlan

    return tuple(chunk.core_stop for chunk in BrainChunkPlan.formal().chunks)


def _validate_partition_stops(
    source_partition_stops: Iterable[int],
    *,
    neuron_count: int,
) -> tuple[int, ...]:
    stops = tuple(int(value) for value in source_partition_stops)
    if not stops:
        raise ValueError("固定路径硬盘整理至少需要一个来源地址分区")
    if stops[-1] != int(neuron_count):
        raise ValueError("最后一个来源地址分区必须恰好覆盖完整神经组织")
    previous = 0
    for value in stops:
        if not previous < value <= int(neuron_count):
            raise ValueError("来源地址分区必须严格递增且处于神经组织范围内")
        previous = value
    return stops


def _require_endpoint_array(endpoints: numpy.ndarray, *, neuron_count: int) -> numpy.ndarray:
    result = numpy.asarray(endpoints)
    if result.dtype != FIXED_PATH_ENDPOINT_DTYPE:
        raise ValueError("固定路径端点记录格式不正确")
    if result.ndim != 1:
        raise ValueError("固定路径端点必须是一维排列")
    if result.size:
        sources = result["source_neuron"]
        targets = result["target_neuron"]
        if numpy.any(sources >= int(neuron_count)) or numpy.any(
            targets >= int(neuron_count)
        ):
            raise ValueError("固定路径端点超出完整神经组织")
        if numpy.any(sources == targets):
            raise ValueError("固定路径不能把同一神经元直接连回自身")
    return result


def _write_json_exclusive(path: Path, payload: dict[str, object]) -> None:
    temporary = path.with_name(path.name + ".partial")
    if temporary.exists():
        raise FileExistsError(f"发现未清理的固定路径整理文件：{temporary.name}")
    with temporary.open("x", encoding="utf-8") as stream:
        json.dump(payload, stream, ensure_ascii=False, indent=2)
        stream.write("\n")
    temporary.replace(path)


def _validate_source_order(
    path: Path,
    *,
    endpoint_count: int,
    neuron_count: int,
) -> None:
    """流式复核整理结果，避免把错误的硬盘顺序当成可执行结构。"""

    if path.stat().st_size != int(endpoint_count) * FIXED_PATH_ENDPOINT_DTYPE.itemsize:
        raise ValueError("固定路径端点硬盘文件尺寸不正确")
    previous_source = -1
    offset = 0
    batch_size = 1_048_576
    with path.open("rb") as stream:
        while offset < endpoint_count:
            count = min(batch_size, int(endpoint_count) - offset)
            values = numpy.fromfile(stream, dtype=FIXED_PATH_ENDPOINT_DTYPE, count=count)
            if values.size != count:
                raise ValueError("固定路径端点硬盘文件意外结束")
            _require_endpoint_array(values, neuron_count=neuron_count)
            sources = values["source_neuron"]
            if sources.size and int(sources[0]) < previous_source:
                raise ValueError("固定路径端点没有按来源神经元地址排列")
            if sources.size > 1 and numpy.any(sources[1:] < sources[:-1]):
                raise ValueError("固定路径端点没有按来源神经元地址排列")
            if sources.size:
                previous_source = int(sources[-1])
            offset += count


def compile_fixed_path_endpoints_by_source(
    directory: Path,
    topology: FixedPathTopology,
    *,
    neuron_count: int = TISSUE_NEURON_COUNT,
    source_partition_stops: Iterable[int] | None = None,
    second_experiment_only: bool = False,
) -> Path:
    """编译已确认固定路径端点，并返回按来源排列的文件位置。

    生成过程中先把每条端点写入其来源神经元所属的硬盘分区；每个分区单独
    排序后才依次合并。因此不会为了数千万条路径把全量端点同时留在
    内存。目标目录必须是新目录，已有结构绝不覆盖。
    """

    root = Path(directory)
    experiment_only = bool(second_experiment_only)
    count = int(
        topology.second_experiment_path_count
        if experiment_only
        else topology.path_count
    )
    if int(neuron_count) <= 0:
        raise ValueError("神经元数量必须大于零")
    if source_partition_stops is None:
        if int(neuron_count) != TISSUE_NEURON_COUNT:
            raise ValueError("非正式大小必须明确提供来源地址分区")
        stops = formal_source_partition_stops()
    else:
        stops = _validate_partition_stops(
            source_partition_stops,
            neuron_count=int(neuron_count),
        )

    endpoint_path = root / ENDPOINT_FILE_NAME
    manifest_path = root / MANIFEST_FILE_NAME
    if root.exists():
        if any(root.iterdir()):
            raise FileExistsError("固定路径端点目录已经存在内容，不能覆盖")
    else:
        root.mkdir(parents=True)
    if endpoint_path.exists() or manifest_path.exists():  # pragma: no cover
        raise FileExistsError("固定路径端点结构已经存在，不能覆盖")

    produced = 0
    final_temporary = root / (ENDPOINT_FILE_NAME + ".partial")
    try:
        with tempfile.TemporaryDirectory(prefix="fixed_path_partition_", dir=root) as scratch_text:
            scratch = Path(scratch_text)
            partition_paths = [
                scratch / f"source_partition_{index:03d}.bin"
                for index in range(len(stops))
            ]
            streams = [path.open("xb") for path in partition_paths]
            try:
                batches = (
                    topology.iter_second_experiment_batches()
                    if experiment_only
                    else topology.iter_batches()
                )
                for batch in batches:
                    if not isinstance(batch, FixedPathEndpointBatch):
                        raise TypeError("固定路径拓扑必须给出端点批次")
                    endpoints = _require_endpoint_array(
                        batch.endpoints,
                        neuron_count=int(neuron_count),
                    )
                    partitions = numpy.searchsorted(
                        numpy.asarray(stops, dtype=numpy.uint32),
                        endpoints["source_neuron"],
                        side="right",
                    )
                    if numpy.any(partitions >= len(stops)):
                        raise ValueError("固定路径来源神经元没有落入任何硬盘分区")
                    for index in numpy.unique(partitions):
                        values = endpoints[partitions == index]
                        values.tofile(streams[int(index)])
                    produced += int(endpoints.size)
            finally:
                for stream in streams:
                    stream.close()

            if produced != count:
                raise ValueError(
                    f"固定路径拓扑实际生成{produced}条端点，但应为{count}条"
                )

            with final_temporary.open("xb") as output:
                for partition_path in partition_paths:
                    values = numpy.fromfile(partition_path, dtype=FIXED_PATH_ENDPOINT_DTYPE)
                    values.sort(order=("source_neuron", "target_neuron"), kind="stable")
                    values.tofile(output)

        _validate_source_order(
            final_temporary,
            endpoint_count=count,
            neuron_count=int(neuron_count),
        )
        final_temporary.replace(endpoint_path)
        _write_json_exclusive(
            manifest_path,
            {
                "format": FORMAT_VERSION,
                "topology_layout": TOPOLOGY_LAYOUT_VERSION,
                "kind": (
                    "第二次实验个体固定路径来源—到达端点结构"
                    if experiment_only
                    else "当前固定路径来源—到达端点结构（含待确认草案）"
                ),
                "neuron_count": int(neuron_count),
                "endpoint_count": count,
                "endpoint_file": ENDPOINT_FILE_NAME,
                "endpoint_dtype": FIXED_PATH_ENDPOINT_DTYPE.descr,
                "source_sorted": True,
                "source_partition_stops": list(stops),
                "contains_path_strength": False,
                "contains_current_activity": False,
                "second_experiment_only": experiment_only,
            },
        )
    except BaseException:
        if final_temporary.exists():
            final_temporary.unlink()
        raise
    return endpoint_path


def open_fixed_path_endpoints_by_source(directory: Path) -> numpy.memmap:
    """只读打开已整理的端点结构，并复核其边界与来源顺序。"""

    root = Path(directory)
    manifest_path = root / MANIFEST_FILE_NAME
    data = json.loads(manifest_path.read_text(encoding="utf-8"))
    if int(data.get("format", -1)) != FORMAT_VERSION:
        raise ValueError("固定路径端点结构版本不受支持")
    if data.get("topology_layout") != TOPOLOGY_LAYOUT_VERSION:
        raise ValueError("固定路径端点结构没有使用当前器官独立继续排列")
    if not bool(data.get("source_sorted", False)):
        raise ValueError("固定路径端点结构没有来源地址顺序")
    if data.get("contains_path_strength") or data.get("contains_current_activity"):
        raise ValueError("固定路径端点结构不能伪装成完整路径状态")
    neuron_count = int(data["neuron_count"])
    endpoint_count = int(data["endpoint_count"])
    path = root / str(data["endpoint_file"])
    _validate_source_order(
        path,
        endpoint_count=endpoint_count,
        neuron_count=neuron_count,
    )
    return numpy.memmap(
        path,
        dtype=FIXED_PATH_ENDPOINT_DTYPE,
        mode="r",
        shape=(endpoint_count,),
    )
