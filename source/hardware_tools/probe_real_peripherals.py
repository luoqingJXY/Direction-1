"""Read-only readiness probe for the real Windows/Minecraft body."""

from __future__ import annotations

import importlib.metadata
import json
from pathlib import Path
import platform

from artificial_brain.windows_peripherals import MinecraftWindowLocator


def main() -> None:
    import pyaudiowpatch as pyaudio

    locator = MinecraftWindowLocator()
    target = locator.locate()
    pa = pyaudio.PyAudio()
    try:
        wasapi_index = int(pa.get_host_api_info_by_type(pyaudio.paWASAPI)["index"])
        outputs = []
        for index in range(pa.get_device_count()):
            info = pa.get_device_info_by_index(index)
            if int(info.get("hostApi", -1)) == wasapi_index and int(info.get("maxOutputChannels", 0)) > 0:
                outputs.append(
                    {
                        "name": str(info["name"]),
                        "channels": int(info["maxOutputChannels"]),
                        "sample_rate": int(round(float(info["defaultSampleRate"]))),
                    }
                )
        loopbacks = [
            {
                "name": str(info["name"]),
                "channels": int(info["maxInputChannels"]),
                "sample_rate": int(round(float(info["defaultSampleRate"]))),
            }
            for info in pa.get_loopback_device_info_generator()
        ]
    finally:
        pa.terminate()

    report = {
        "platform": platform.platform(),
        "dependencies": {
            "mss": importlib.metadata.version("mss"),
            "PyAudioWPatch": importlib.metadata.version("PyAudioWPatch"),
        },
        "minecraft_window": None
        if target is None
        else {
            "title": target.title,
            "process": target.process_name,
            "client_size": [target.width, target.height],
            "foreground": target.foreground,
        },
        "output_devices": outputs,
        "loopback_devices": loopbacks,
        "ready_for_real_capture": target is not None and bool(loopbacks),
        "operator_choices_still_required": [
            "route Minecraft alone to one listed output/loopback device",
            "copy that exact loopback name and the vocal output name into real_experiment.toml",
            "provide two-host TLS certificates and transport secret",
            "leave motor.armed=false until the dry run has passed",
        ],
    }
    destination = Path("artifacts/peripheral_commission/real_environment_probe.json")
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))
    print(destination.resolve())


if __name__ == "__main__":
    main()
