"""把已经闭合的第二次实验人工DNA一次编译成第一次出生硬盘个体。"""

from __future__ import annotations

from dataclasses import asdict
from itertools import islice
import json
from pathlib import Path
import shutil
import tempfile
from typing import Callable

import numpy

from .brain_address_plan import build_brain_address_plan
from .brain_birth_readiness import build_brain_birth_readiness
from .brain_geometry import TISSUE_NEURON_COUNT
from .brain_storage import (
    DiskBackedBrainState,
    EMOTION_CONTROL_PATH_DTYPE,
    FIXED_PATH_DTYPE,
    BrainStorageSpec,
)
from .fixed_path_compilation import formal_source_partition_stops
from .fixed_path_topology import (
    SECOND_EXPERIMENT_ACTIVE_FIXED_PATH_COUNT,
    FixedPathTopology,
)
from .formal_brain_topology import FormalBrainTopology
from .formal_local_path_birth import (
    FORMAL_LOCAL_FORMATION_PERMISSION_COUNT,
    FORMAL_LOCAL_PATH_CHANGE_RATE,
    FORMAL_LOCAL_PATH_FORMATION_THRESHOLD,
    FORMAL_SLEEP_PATH_WEAKENING,
    FormalLocalPathBirth,
    iter_path_response_record_batches,
)
from .formal_neuron_birth_values import FormalNeuronBirthValues
from .neuron_nature_topology import NeuronNatureTopology
from .organ_entrances import SECOND_EXPERIMENT_ENTRANCES
from .occurrence_workspace import DiskBackedOccurrenceWorkspace
from .output_control_paths import (
    KNOWN_OUTPUT_CONTROL_PATH_COUNT,
    build_known_output_control_path_genes_by_source,
)
from .vital_state_instinct import build_vital_state_instinct
from .vital_state_reference_plan import load_compiled_vital_state_plan


FORMAL_EMOTION_CONTROL_PATH_COUNT = 70
FORMAL_NEURON_MODULATION_RESPONSE_COUNT = 0
COMPILATION_STATUS_FILE = "formal_birth_compilation_status.json"
_WORKING_DISK_MARGIN_BYTES = 2_000_000_000


def _report(progress: Callable[[str], None] | None, text: str) -> None:
    if progress is not None:
        progress(text)


def _write_status(root: Path, stage: str, **values: object) -> None:
    path = root / COMPILATION_STATUS_FILE
    temporary = root / (COMPILATION_STATUS_FILE + ".partial")
    payload = {"stage": stage, **values}
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    temporary.replace(path)


def _existing_parent(path: Path) -> Path:
    current = path.resolve()
    while not current.exists():
        parent = current.parent
        if parent == current:
            raise FileNotFoundError("找不到可用于建立人工生命个体的上级目录")
        current = parent
    return current


def _compile_fixed_paths(
    state: DiskBackedBrainState,
    topology: FixedPathTopology,
    *,
    progress: Callable[[str], None] | None,
) -> None:
    stops = numpy.asarray(formal_source_partition_stops(), dtype="<u4")
    output = state.open_fixed_paths("r+")
    if output.size != SECOND_EXPERIMENT_ACTIVE_FIXED_PATH_COUNT:
        raise RuntimeError("正式固定路径文件数量与本次实验出生范围不一致")

    produced = 0
    with tempfile.TemporaryDirectory(
        prefix="formal_fixed_path_",
        dir=state.directory,
    ) as scratch_text:
        scratch = Path(scratch_text)
        paths = [scratch / f"partition_{index:03d}.bin" for index in range(stops.size)]
        streams = [path.open("xb") for path in paths]
        strength_iterator = None
        strength_family = None
        try:
            for batch in topology.iter_second_experiment_batches():
                if batch.family != strength_family:
                    if strength_iterator is not None:
                        try:
                            next(strength_iterator)
                        except StopIteration:
                            pass
                        else:
                            raise RuntimeError("上一类固定路径强度多于其端点数量")
                    strength_family = batch.family
                    strength_iterator = topology.iter_confirmed_strengths(batch.family)
                strengths = numpy.asarray(
                    tuple(islice(strength_iterator, int(batch.endpoints.size))),
                    dtype="<f4",
                )
                if strengths.size != batch.endpoints.size:
                    raise RuntimeError("固定路径端点与出生强度没有逐条对应")
                records = numpy.zeros(batch.endpoints.size, dtype=FIXED_PATH_DTYPE)
                records["source_neuron"] = batch.endpoints["source_neuron"]
                records["target_neuron"] = batch.endpoints["target_neuron"]
                records["path_strength"] = strengths
                partitions = numpy.searchsorted(
                    stops,
                    records["source_neuron"],
                    side="right",
                )
                if numpy.any(partitions >= stops.size):
                    raise RuntimeError("固定路径来源没有落入正式运行分块")
                for index in numpy.unique(partitions):
                    records[partitions == index].tofile(streams[int(index)])
                produced += int(records.size)
        finally:
            for stream in streams:
                stream.close()

        if produced != output.size:
            raise RuntimeError("正式固定路径没有完整写入出生个体")
        offset = 0
        previous_source = -1
        for index, path in enumerate(paths):
            records = numpy.fromfile(path, dtype=FIXED_PATH_DTYPE)
            records.sort(order=("source_neuron", "target_neuron"), kind="stable")
            if records.size:
                if int(records["source_neuron"][0]) < previous_source:
                    raise RuntimeError("正式固定路径没有保持来源地址顺序")
                previous_source = int(records["source_neuron"][-1])
            output[offset : offset + records.size] = records
            offset += int(records.size)
            _report(progress, f"固定路径来源分区 {index + 1}/{len(paths)} 已写入")
        if offset != output.size:
            raise RuntimeError("正式固定路径整理后的记录数发生变化")
    output.flush()


def _compile_emotion_control_paths(
    state: DiskBackedBrainState,
    topology: FixedPathTopology,
) -> None:
    output = state.open_emotion_control_paths("r+")
    records = numpy.zeros(
        topology.emotion_control_path_count,
        dtype=EMOTION_CONTROL_PATH_DTYPE,
    )
    for index, (source, entrance, strength) in enumerate(
        topology.iter_emotion_control_paths()
    ):
        records[index] = source, entrance, strength, 0.0
    records.sort(order=("source_neuron", "control_entrance"), kind="stable")
    output[:] = records
    output.flush()


def _compile_output_control_paths(state: DiskBackedBrainState) -> None:
    genes = build_known_output_control_path_genes_by_source()
    output = state.open_output_control_paths("r+")
    output["source_neuron"] = genes["source_neuron"]
    output["control_entrance"] = genes["control_entrance"]
    output["path_strength"] = genes["path_strength"]
    output["current_activity"] = 0.0
    output.flush()


def _compile_neurons(
    state: DiskBackedBrainState,
    brain_topology: FormalBrainTopology,
    fixed_paths: FixedPathTopology,
    *,
    progress: Callable[[str], None] | None,
) -> None:
    arrays = state.open_arrays("r+")
    brain_topology.apply_nature_to(arrays["neuron_nature"])
    arrays["neuron_nature"].flush()
    _report(progress, "1.024亿个神经元性质已写入")
    for number, batch in enumerate(
        FormalNeuronBirthValues(brain_topology, fixed_paths).iter_batches(),
        start=1,
    ):
        arrays["neuron_response_gain"][batch.start : batch.stop] = batch.values[
            "response_gain"
        ]
        arrays["neuron_threshold"][batch.start : batch.stop] = batch.values[
            "threshold"
        ]
        if number % 32 == 0:
            _report(progress, f"神经元出生值已写入 {batch.stop}/{TISSUE_NEURON_COUNT}")
    arrays["neuron_response_gain"].flush()
    arrays["neuron_threshold"].flush()


def _compile_local_paths_and_responses(
    state: DiskBackedBrainState,
    birth: FormalLocalPathBirth,
    instinct,
    *,
    progress: Callable[[str], None] | None,
) -> None:
    arrays = state.open_arrays("r+")
    for batch in birth.iter_batches(chunk_depth=1):
        start = batch.start_index
        stop = batch.stop_index
        arrays["local_path_formation_mask"][start:stop] = batch.formation_masks.reshape(-1)
        for direction in range(26):
            permitted = (
                batch.formation_masks.reshape(-1)
                & numpy.uint32(1 << direction)
            ) != 0
            rates = arrays["local_path_change_rate"][start:stop, direction]
            rates[:] = 0.0
            rates[permitted] = FORMAL_LOCAL_PATH_CHANGE_RATE
        # 零是已经明确的出生形成阈值；全文件在建立时已经物理置零。
        if FORMAL_LOCAL_PATH_FORMATION_THRESHOLD != 0.0:  # pragma: no cover
            arrays["local_path_formation_threshold"][start:stop] = (
                FORMAL_LOCAL_PATH_FORMATION_THRESHOLD
            )
        _report(progress, f"相邻路径出生值已写入深度层 {batch.stop_z}/{160}")
    arrays["local_path_formation_mask"].flush()
    arrays["local_path_change_rate"].flush()
    arrays["local_path_formation_threshold"].flush()

    responses = state.open_path_modulation_responses("r+")
    offset = 0
    for depth, records in enumerate(
        iter_path_response_record_batches(birth, instinct, chunk_depth=1),
        start=1,
    ):
        responses[offset : offset + records.size] = records
        offset += int(records.size)
        _report(progress, f"70路路径物质受体已写入深度层 {depth}/{160}")
    if offset != FORMAL_LOCAL_FORMATION_PERMISSION_COUNT:
        raise RuntimeError("路径物质受体没有逐条覆盖全部允许形成关系")
    responses.flush()


def compile_formal_second_experiment_birth(
    directory: Path,
    *,
    progress: Callable[[str], None] | None = print,
) -> dict[str, object]:
    """在全新目录中生成正式第一次出生状态；任何已有目录都不会覆盖。"""

    root = Path(directory)
    sleep_weakening = float(FORMAL_SLEEP_PATH_WEAKENING)
    if root.exists():
        raise FileExistsError("正式个体目录已经存在；人工DNA出生不能覆盖已有个体")

    entrances = SECOND_EXPERIMENT_ENTRANCES
    instinct = build_vital_state_instinct(
        load_compiled_vital_state_plan(),
        tissue_start_index=entrances.next_free_index,
    )
    fixed_paths = FixedPathTopology(instinct, entrances=entrances)
    addresses = build_brain_address_plan(instinct, entrances=entrances)
    readiness = build_brain_birth_readiness(
        entrances=entrances,
        instinct=instinct,
        addresses=addresses,
        topology=fixed_paths,
    )
    readiness.require_ready()

    spec = BrainStorageSpec.formal()
    total_bytes = spec.total_byte_count(
        SECOND_EXPERIMENT_ACTIVE_FIXED_PATH_COUNT,
        FORMAL_EMOTION_CONTROL_PATH_COUNT,
        KNOWN_OUTPUT_CONTROL_PATH_COUNT,
        FORMAL_NEURON_MODULATION_RESPONSE_COUNT,
        FORMAL_LOCAL_FORMATION_PERMISSION_COUNT,
    )
    occurrence_workspace_bytes = DiskBackedOccurrenceWorkspace.required_byte_count(
        spec,
        fixed_path_count=SECOND_EXPERIMENT_ACTIVE_FIXED_PATH_COUNT,
        emotion_control_path_count=FORMAL_EMOTION_CONTROL_PATH_COUNT,
        output_control_path_count=KNOWN_OUTPUT_CONTROL_PATH_COUNT,
    )
    available = shutil.disk_usage(_existing_parent(root)).free
    peak_required = total_bytes + occurrence_workspace_bytes + _WORKING_DISK_MARGIN_BYTES
    if available < peak_required:
        raise OSError(
            f"正式个体及一次完整发生需要至少{peak_required / 1e9:.2f} GB"
            "可用空间（含工作区与编译余量）"
        )

    state = DiskBackedBrainState(root, spec)
    state.create(
        fixed_path_count=SECOND_EXPERIMENT_ACTIVE_FIXED_PATH_COUNT,
        emotion_control_path_count=FORMAL_EMOTION_CONTROL_PATH_COUNT,
        output_control_path_count=KNOWN_OUTPUT_CONTROL_PATH_COUNT,
        neuron_modulation_response_count=FORMAL_NEURON_MODULATION_RESPONSE_COUNT,
        path_modulation_response_count=FORMAL_LOCAL_FORMATION_PERMISSION_COUNT,
        birth_readiness=readiness,
    )
    birth_manifest = state._manifest()
    birth_manifest["sleep_path_weakening"] = sleep_weakening
    state.manifest_path.write_text(
        json.dumps(birth_manifest, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    _write_status(root, "files_created", total_bytes=total_bytes)

    brain_topology = FormalBrainTopology(
        NeuronNatureTopology(instinct, entrances=entrances)
    )
    local_birth = FormalLocalPathBirth(brain_topology)
    ledger = local_birth.build_ledger()
    _write_status(root, "topology_checked", local_path_ledger=asdict(ledger))

    _compile_fixed_paths(state, fixed_paths, progress=progress)
    _write_status(root, "fixed_paths_written")
    _compile_emotion_control_paths(state, fixed_paths)
    _compile_output_control_paths(state)
    _write_status(root, "organ_control_paths_written")
    _compile_neurons(state, brain_topology, fixed_paths, progress=progress)
    _write_status(root, "neurons_written")
    _compile_local_paths_and_responses(
        state,
        local_birth,
        instinct,
        progress=progress,
    )
    _write_status(root, "complete", total_bytes=total_bytes)
    state.validate_files()
    return {
        "ready": True,
        "directory": str(root.resolve()),
        "neuron_count": TISSUE_NEURON_COUNT,
        "fixed_path_count": SECOND_EXPERIMENT_ACTIVE_FIXED_PATH_COUNT,
        "local_path_count": ledger.classified_count,
        "path_modulation_response_count": FORMAL_LOCAL_FORMATION_PERMISSION_COUNT,
        "total_bytes": total_bytes,
        "occurrence_workspace_bytes": occurrence_workspace_bytes,
        "peak_required_bytes": peak_required,
        "sleep_path_weakening": sleep_weakening,
    }
