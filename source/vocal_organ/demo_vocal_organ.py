"""Render a direct, non-text demonstration of the standalone vocal organ."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

from artificial_brain.vocal import (
    CONSONANT_UNITS,
    VOWEL_UNITS,
    RecordedVowelBackend,
    VocalControlSignal,
    VocalOrgan,
    write_pcm16_wav,
)


def _activities(names: tuple[str, ...], values: dict[str, float]) -> tuple[float, ...]:
    return tuple(values.get(name, 0.0) for name in names)


def build_demonstration(block_seconds: float) -> list[VocalControlSignal]:
    controls: list[VocalControlSignal] = []

    def add(
        seconds: float,
        vowels: dict[str, float] | None = None,
        consonants: dict[str, float] | None = None,
        *,
        gate: float = 1.0,
        pitch: float = 0.63,
        energy: float = 0.68,
        airflow: float = 0.18,
        tension: float = 0.68,
    ) -> None:
        for _ in range(max(1, round(seconds / block_seconds))):
            controls.append(
                VocalControlSignal(
                    tick=len(controls),
                    gate=gate,
                    vowel_activity=_activities(VOWEL_UNITS, vowels or {}),
                    consonant_activity=_activities(CONSONANT_UNITS, consonants or {}),
                    pitch=pitch,
                    energy=energy,
                    airflow=airflow,
                    tension=tension,
                )
            )

    add(0.20, gate=0.0, energy=0.0)
    for index, vowel in enumerate(("a", "i", "u", "e", "o", "y")):
        add(0.42, {vowel: 1.0}, pitch=0.58 + 0.025 * index)
        add(0.08, {vowel: 0.2}, gate=0.0, energy=0.0)

    # Continuous blends demonstrate that the organ does not select one winner.
    blend_steps = 24
    for index in range(blend_steps):
        ratio = index / (blend_steps - 1)
        add(
            block_seconds,
            {"a": 1.0 - ratio, "i": ratio},
            pitch=0.66 + 0.04 * ratio,
            energy=0.64,
        )
    add(0.12, gate=0.0, energy=0.0)

    # Short consonant activity overlays vowel material; there is still no text input.
    for consonant, vowel in (("m", "a"), ("sh", "i"), ("k", "u"), ("s", "e"), ("n", "o")):
        add(0.08, {vowel: 0.65}, {consonant: 1.0}, energy=0.70, airflow=0.28)
        add(0.30, {vowel: 1.0}, energy=0.70)
        add(0.07, {vowel: 0.2}, gate=0.0, energy=0.0)
    add(0.30, gate=0.0, energy=0.0)
    return controls


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("artifacts/vocal_organ_demo.wav"),
    )
    parser.add_argument(
        "--feedback",
        type=Path,
        default=Path("artifacts/vocal_organ_demo_feedback.csv"),
    )
    arguments = parser.parse_args()
    material_directory = Path("artifacts/vocal_material")
    organ = VocalOrgan(
        backend=RecordedVowelBackend.from_directory(material_directory),
        block_seconds=0.020,
    )
    controls = build_demonstration(organ.block_seconds)
    samples, feedback = organ.render(controls)
    write_pcm16_wav(arguments.output, samples, organ.sample_rate)
    arguments.feedback.parent.mkdir(parents=True, exist_ok=True)
    rows = []
    for current_control, current_feedback in zip(controls, feedback):
        row = {
            "tick": current_control.tick,
            "gate": current_control.gate,
            "pitch_control": current_control.pitch,
            "energy_control": current_control.energy,
            "airflow_control": current_control.airflow,
            "tension_control": current_control.tension,
            "rms": current_feedback.rms,
            "peak": current_feedback.peak,
            "f0_hz": current_feedback.f0_hz,
        }
        row.update(
            {
                f"control_vowel_{name}": value
                for name, value in zip(VOWEL_UNITS, current_control.vowel_activity)
            }
        )
        row.update(
            {
                f"control_consonant_{name}": value
                for name, value in zip(CONSONANT_UNITS, current_control.consonant_activity)
            }
        )
        row.update(
            {
                f"organ_activity_{index}": value
                for index, value in enumerate(current_feedback.organ_activity)
            }
        )
        row.update(
            {
                f"spectrum_{index}": value
                for index, value in enumerate(current_feedback.spectrum_activity)
            }
        )
        row.update(
            {
                f"spectrum_change_{index}": value
                for index, value in enumerate(current_feedback.spectrum_change)
            }
        )
        rows.append(row)
    with arguments.feedback.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    duration = len(samples) / organ.sample_rate
    peak = max((item.peak for item in feedback), default=0.0)
    rms = max((item.rms for item in feedback), default=0.0)
    print(f"backend={organ.backend.name}")
    print(f"sample_rate={organ.sample_rate}")
    print(f"duration_seconds={duration:.3f}")
    print(f"peak={peak:.6f}")
    print(f"max_frame_rms={rms:.6f}")
    print(f"output={arguments.output.resolve()}")
    print(f"feedback={arguments.feedback.resolve()}")


if __name__ == "__main__":
    main()
