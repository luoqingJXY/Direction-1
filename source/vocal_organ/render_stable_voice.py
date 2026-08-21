"""Render one isolated, settled vocal-organ sound.

The organ is conditioned while its acoustic gate is closed.  When the gate
opens, pitch, energy, airflow, tension, vowel selection, and consonant activity
are already constant; only the physical amplitude envelope is allowed to move.
"""

from __future__ import annotations

import csv
from pathlib import Path

from artificial_brain.vocal import (
    CONSONANT_UNITS,
    MORPHOLOGY_CONTROLS,
    VOWEL_UNITS,
    OneShotRecordedVowelBackend,
    VocalControlSignal,
    VocalOrgan,
    write_pcm16_wav,
)


BLOCK_SECONDS = 0.020


def _stable_control(tick: int, gate: float) -> VocalControlSignal:
    morphology = {
        "tract_length": 0.38,
        "glottal_depth": 0.60,
        "brightness": 0.68,
        "breathiness": 0.10,
        "roughness": 0.04,
        "nasality": 0.08,
        "resonance": 0.66,
    }
    return VocalControlSignal(
        tick=tick,
        gate=gate,
        vowel_activity=tuple(1.0 if vowel == "a" else 0.0 for vowel in VOWEL_UNITS),
        consonant_activity=(0.0,) * len(CONSONANT_UNITS),
        pitch=0.63,
        energy=0.62,
        airflow=0.06,
        tension=0.64,
        morphology_activity=tuple(morphology[name] for name in MORPHOLOGY_CONTROLS),
    )


def _extend(controls: list[VocalControlSignal], seconds: float, gate: float) -> None:
    for _ in range(round(seconds / BLOCK_SECONDS)):
        controls.append(_stable_control(len(controls), gate))


def main() -> None:
    controls: list[VocalControlSignal] = []
    _extend(controls, 0.30, 0.0)  # settle all non-acoustic organ state
    _extend(controls, 0.20, 1.0)  # one opening; source material is never looped
    _extend(controls, 0.80, 0.0)  # natural release followed by real silence

    organ = VocalOrgan(
        backend=OneShotRecordedVowelBackend.from_directory("artifacts/vocal_material"),
        block_seconds=BLOCK_SECONDS,
    )
    samples, feedback = organ.render(controls)
    output = Path("artifacts/stable_voice_a.wav")
    write_pcm16_wav(output, samples, organ.sample_rate)

    feedback_path = Path("artifacts/stable_voice_a_feedback.csv")
    with feedback_path.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.writer(handle)
        writer.writerow(("tick", "gate", "rms", "peak", "f0_hz", "envelope"))
        for control, state in zip(controls, feedback):
            writer.writerow(
                (
                    control.tick,
                    control.gate,
                    state.rms,
                    state.peak,
                    state.f0_hz,
                    state.organ_activity[0],
                )
            )

    print(output.resolve())
    print(feedback_path.resolve())


if __name__ == "__main__":
    main()
