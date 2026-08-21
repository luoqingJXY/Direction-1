from __future__ import annotations

import argparse
from copy import deepcopy
import json
from pathlib import Path
import time

import numpy as np

from artificial_brain.brain import ArtificialBrain
from artificial_brain.hardware_backend import (
    DeviceBackendUnavailable,
    NumpyCPUBackend,
    PackedActiveStructure,
    TorchROCmBackend,
)


def _synthetic_numpy_case(neurons: int, paths: int, rounds: int) -> dict[str, float | int]:
    rng = np.random.default_rng(neurons + paths)
    activity = rng.uniform(0.0, 0.3, neurons).astype(np.float32)
    threshold = rng.uniform(0.01, 0.12, neurons).astype(np.float32)
    recovery = rng.uniform(0.4, 0.65, neurons).astype(np.float32)
    source = rng.integers(0, neurons, paths, dtype=np.int32)
    target_index = rng.integers(0, neurons, paths, dtype=np.int32)
    tendency = rng.uniform(0.01, 0.3, paths).astype(np.float32)
    inhibition = rng.uniform(0.0, 0.08, paths).astype(np.float32)
    started = time.perf_counter()
    for _ in range(rounds):
        transmitted = activity[source] * tendency * (1.0 - inhibition)
        incoming = np.bincount(target_index, weights=transmitted, minlength=neurons).astype(
            np.float32, copy=False
        )
        normalized = 1.0 - np.exp(-incoming)
        desired = np.clip(
            (normalized - threshold) / np.maximum(1e-9, 1.0 - threshold), 0.0, 1.0
        )
        mask = incoming > 1e-12
        blend = np.clip(recovery * 0.16, 0.01, 0.18)
        activity[mask] = np.clip(
            activity[mask] + blend[mask] * (desired[mask] - activity[mask]), 0.0, 1.0
        )
    seconds = time.perf_counter() - started
    transmissions = paths * rounds
    return {
        "neurons": neurons,
        "paths": paths,
        "rounds": rounds,
        "seconds": seconds,
        "million_path_transmissions_per_second": transmissions / seconds / 1_000_000,
        "estimated_four_round_loops_per_second": transmissions / seconds / (paths * 4),
    }


def _synthetic_gpu_case(
    backend: TorchROCmBackend, neurons: int, paths: int, rounds: int
) -> dict[str, float | int]:
    torch = backend.torch
    device = backend.device
    torch.manual_seed(neurons + paths)
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    activity = torch.rand(neurons, device=device, dtype=torch.float32) * 0.3
    threshold = torch.rand(neurons, device=device, dtype=torch.float32) * 0.11 + 0.01
    recovery = torch.rand(neurons, device=device, dtype=torch.float32) * 0.25 + 0.4
    source = torch.randint(0, neurons, (paths,), device=device, dtype=torch.int64)
    target_index = torch.randint(0, neurons, (paths,), device=device, dtype=torch.int64)
    tendency = torch.rand(paths, device=device, dtype=torch.float32) * 0.29 + 0.01
    inhibition = torch.rand(paths, device=device, dtype=torch.float32) * 0.08
    device_state = {
        "activity": activity,
        "threshold": threshold,
        "recovery": recovery,
        "source_index": source,
        "target_index": target_index,
        "tendency": tendency,
        "inhibition": inhibition,
    }
    for _ in range(20):
        backend.propagate_round(device_state)
    torch.cuda.synchronize()
    started = time.perf_counter()
    for _ in range(rounds):
        backend.propagate_round(device_state)
    torch.cuda.synchronize()
    seconds = time.perf_counter() - started
    transmissions = paths * rounds
    allocated = torch.cuda.max_memory_allocated()
    reserved = torch.cuda.max_memory_reserved()
    result = {
        "neurons": neurons,
        "paths": paths,
        "rounds": rounds,
        "seconds": seconds,
        "million_path_transmissions_per_second": transmissions / seconds / 1_000_000,
        "estimated_four_round_loops_per_second": transmissions / seconds / (paths * 4),
        "peak_allocated_mb": allocated / 1_048_576,
        "peak_reserved_mb": reserved / 1_048_576,
    }
    del device_state, activity, threshold, recovery, source, target_index, tendency, inhibition
    torch.cuda.empty_cache()
    return result


def benchmark(rounds: int = 2_000) -> dict[str, object]:
    rng = np.random.default_rng(31)
    original = ArtificialBrain()
    initial = rng.uniform(0.0, 0.35, len(original.neurons)).astype(np.float32)
    for value, neuron in zip(initial, original.neurons.values()):
        neuron.activity = float(value)

    packed = PackedActiveStructure.from_brain(original)
    object_brain = deepcopy(original)
    object_started = time.perf_counter()
    for _ in range(rounds):
        object_brain._propagate_round(sleeping=False)
    object_seconds = time.perf_counter() - object_started

    numpy_started = time.perf_counter()
    for _ in range(rounds):
        NumpyCPUBackend.propagate_round(packed)
    numpy_seconds = time.perf_counter() - numpy_started

    result: dict[str, object] = {
        "neurons": len(original.neurons),
        "paths": len(original.paths),
        "rounds": rounds,
        "python_object_seconds": object_seconds,
        "numpy_packed_seconds": numpy_seconds,
        "numpy_speedup": object_seconds / numpy_seconds,
        "packed_bytes": packed.byte_size,
        "packed_bytes_per_neuron_and_path": packed.byte_size
        / (len(original.neurons) + len(original.paths)),
        "gpu_available": False,
        "numpy_scale_cases": [
            _synthetic_numpy_case(2_500, 30_000, 500),
            _synthetic_numpy_case(50_000, 1_000_000, 30),
        ],
    }
    try:
        backend = TorchROCmBackend()
        gpu_source = PackedActiveStructure.from_brain(original)
        device_state = backend.pack(gpu_source)
        for _ in range(50):
            backend.propagate_round(device_state)
        backend.torch.cuda.synchronize()
        gpu_started = time.perf_counter()
        for _ in range(rounds):
            backend.propagate_round(device_state)
        backend.torch.cuda.synchronize()
        gpu_seconds = time.perf_counter() - gpu_started
        result.update(
            {
                "gpu_available": True,
                "gpu_name": backend.torch.cuda.get_device_name(0),
                "gpu_seconds": gpu_seconds,
                "gpu_speedup_vs_objects": object_seconds / gpu_seconds,
                "gpu_scale_cases": [
                    _synthetic_gpu_case(backend, 2_500, 30_000, 5_000),
                    _synthetic_gpu_case(backend, 50_000, 1_000_000, 300),
                    _synthetic_gpu_case(backend, 200_000, 10_000_000, 50),
                ],
                "cpu_gpu_crossover_cases": [
                    {
                        "numpy": _synthetic_numpy_case(neurons, paths, 100),
                        "gpu": _synthetic_gpu_case(backend, neurons, paths, 1_000),
                    }
                    for neurons, paths in (
                        (5_000, 100_000),
                        (7_500, 150_000),
                        (10_000, 200_000),
                        (15_000, 300_000),
                    )
                ],
            }
        )
        cpu_check = PackedActiveStructure.from_brain(original)
        NumpyCPUBackend.propagate_round(cpu_check)
        gpu_check = backend.pack(PackedActiveStructure.from_brain(original))
        backend.propagate_round(gpu_check)
        backend.torch.cuda.synchronize()
        gpu_activity = gpu_check["activity"].cpu().numpy()
        result["cpu_gpu_one_round_max_abs_difference"] = float(
            np.max(np.abs(cpu_check.activity - gpu_activity))
        )
    except DeviceBackendUnavailable as exc:
        result["gpu_unavailable_reason"] = str(exc)
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description="CPU/RAM/ROCm packed structure benchmark")
    parser.add_argument("--rounds", type=int, default=2_000)
    parser.add_argument("--output", type=Path, default=Path("artifacts/hardware_benchmark.json"))
    args = parser.parse_args()
    result = benchmark(args.rounds)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
