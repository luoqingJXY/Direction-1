"""Render isolated vocal-organ units: one unit, one reset, one WAV file."""

from __future__ import annotations

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


BLOCK_SECONDS = 0.020


def _one(names: tuple[str, ...], active: str | None) -> tuple[float, ...]:
    return tuple(1.0 if name == active else 0.0 for name in names)


def _control(
    tick: int,
    vowel: str | None,
    consonant: str | None = None,
    *,
    gate: float = 1.0,
) -> VocalControlSignal:
    return VocalControlSignal(
        tick=tick,
        gate=gate,
        vowel_activity=_one(VOWEL_UNITS, vowel),
        consonant_activity=_one(CONSONANT_UNITS, consonant),
        pitch=0.63,
        energy=0.66 if gate else 0.0,
        airflow=0.16,
        tension=0.66,
    )


def _repeat(
    controls: list[VocalControlSignal],
    seconds: float,
    vowel: str | None,
    consonant: str | None = None,
    *,
    gate: float = 1.0,
) -> None:
    for _ in range(max(1, round(seconds / BLOCK_SECONDS))):
        controls.append(_control(len(controls), vowel, consonant, gate=gate))


def _unit_controls(vowel: str, consonant: str | None) -> list[VocalControlSignal]:
    controls: list[VocalControlSignal] = []
    _repeat(controls, 0.20, None, gate=0.0)
    if consonant is not None:
        _repeat(controls, 0.08, vowel, consonant)
    _repeat(controls, 0.62, vowel)
    # Let the organ's own 65 ms release settle naturally, then retain a
    # genuinely silent tail.  This prevents adjacent previews in a media
    # player from sounding as if two otherwise separate units were joined.
    _repeat(controls, 0.80, None, gate=0.0)
    return controls


def main() -> None:
    material_directory = Path("artifacts/vocal_material")
    output_directory = Path("artifacts/vocal_units")
    output_directory.mkdir(parents=True, exist_ok=True)
    definitions = [
        *[(f"vowel_{vowel}", vowel, None) for vowel in VOWEL_UNITS],
        ("syllable_ma", "a", "m"),
        ("syllable_shi", "i", "sh"),
        ("syllable_ku", "u", "k"),
        ("syllable_se", "e", "s"),
        ("syllable_no", "o", "n"),
    ]
    feedback_rows: list[dict[str, float | int | str]] = []
    for name, vowel, consonant in definitions:
        organ = VocalOrgan(
            backend=RecordedVowelBackend.from_directory(material_directory),
            block_seconds=BLOCK_SECONDS,
        )
        controls = _unit_controls(vowel, consonant)
        samples, feedback = organ.render(controls)
        destination = output_directory / f"{name}.wav"
        write_pcm16_wav(destination, samples, organ.sample_rate)
        for current_control, current_feedback in zip(controls, feedback):
            feedback_rows.append(
                {
                    "file": destination.name,
                    "tick": current_control.tick,
                    "vowel": vowel,
                    "consonant": consonant or "",
                    "gate": current_control.gate,
                    "rms": current_feedback.rms,
                    "peak": current_feedback.peak,
                    "f0_hz": current_feedback.f0_hz,
                }
            )
        print(destination.resolve())
    feedback_path = output_directory / "feedback.csv"
    with feedback_path.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(feedback_rows[0]))
        writer.writeheader()
        writer.writerows(feedback_rows)
    print(feedback_path.resolve())


if __name__ == "__main__":
    main()
