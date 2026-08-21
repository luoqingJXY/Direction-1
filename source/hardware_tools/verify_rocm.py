from __future__ import annotations

import argparse
import json
from pathlib import Path
import platform
import time

import numpy as np
import torch


def verify() -> dict[str, object]:
    available = torch.cuda.is_available()
    result: dict[str, object] = {
        "python": platform.python_version(),
        "torch": torch.__version__,
        "rocm": torch.version.hip,
        "numpy": np.__version__,
        "gpu_available": available,
        "device_count": torch.cuda.device_count(),
    }
    if not available:
        return result

    device = torch.device("cuda")
    free, total = torch.cuda.mem_get_info(0)
    source = torch.tensor([1.0, 2.0, 3.0], device=device)
    target = torch.tensor([0, 1, 0], device=device)
    aggregate = torch.zeros(2, device=device)
    aggregate.index_add_(0, target, source)
    torch.cuda.synchronize()

    transfer_values = np.ones(64 * 1024 * 1024, dtype=np.float32)  # 256 MiB
    transfer_started = time.perf_counter()
    device_values = torch.from_numpy(transfer_values).to(device)
    torch.cuda.synchronize()
    upload_seconds = time.perf_counter() - transfer_started
    download_started = time.perf_counter()
    returned = device_values.cpu().numpy()
    torch.cuda.synchronize()
    download_seconds = time.perf_counter() - download_started
    transfer_bytes = transfer_values.nbytes

    result.update(
        {
            "gpu_name": torch.cuda.get_device_name(0),
            "vram_total_bytes": total,
            "vram_free_bytes_before_test": free,
            "index_add_result": aggregate.cpu().tolist(),
            "index_add_passed": aggregate.cpu().tolist() == [4.0, 2.0],
            "upload_256mib_seconds": upload_seconds,
            "upload_gib_per_second": transfer_bytes / upload_seconds / (1024**3),
            "download_256mib_seconds": download_seconds,
            "download_gib_per_second": transfer_bytes / download_seconds / (1024**3),
            "round_trip_equal": bool(np.array_equal(transfer_values, returned)),
        }
    )
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description="Verify AMD ROCm for the artificial-life runtime")
    parser.add_argument("--output", type=Path, default=Path("artifacts/rocm_environment.json"))
    args = parser.parse_args()
    result = verify()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))
    return 0 if result.get("gpu_available") and result.get("index_add_passed") else 1


if __name__ == "__main__":
    raise SystemExit(main())

