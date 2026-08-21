"""Measure whether every vocal morphology control expands acoustic output."""

from __future__ import annotations

import csv
from pathlib import Path

import numpy as np

from artificial_brain.vocal import (
    CONSONANT_UNITS,
    MORPHOLOGY_CONTROLS,
    MORPHOLOGY_NEUTRAL,
    VOWEL_UNITS,
    OneShotRecordedVowelBackend,
    VocalControlSignal,
    VocalOrgan,
)


BLOCK_SECONDS = 0.020


def _control(tick: int, gate: float, morphology: tuple[float, ...]) -> VocalControlSignal:
    return VocalControlSignal(
        tick=tick,
        gate=gate,
        vowel_activity=tuple(1.0 if name == "a" else 0.0 for name in VOWEL_UNITS),
        consonant_activity=(0.0,) * len(CONSONANT_UNITS),
        pitch=0.63,
        energy=0.62,
        airflow=0.06,
        tension=0.64,
        morphology_activity=morphology,
    )


def _render(morphology: tuple[float, ...]) -> tuple[np.ndarray, float, int]:
    controls = [
        *[_control(tick, 0.0, morphology) for tick in range(15)],
        *[_control(tick, 1.0, morphology) for tick in range(15, 25)],
        *[_control(tick, 0.0, morphology) for tick in range(25, 35)],
    ]
    organ = VocalOrgan(
        backend=OneShotRecordedVowelBackend.from_directory("artifacts/vocal_material"),
        block_seconds=BLOCK_SECONDS,
    )
    samples, feedback = organ.render(controls)
    start = 15 * organ.sample_count
    end = 25 * organ.sample_count
    return samples[start:end].astype(np.float64), feedback[24].f0_hz, organ.sample_rate


def _spectrum(samples: np.ndarray) -> np.ndarray:
    magnitude = np.abs(np.fft.rfft(samples * np.hanning(len(samples))))
    return magnitude / max(1e-12, float(np.linalg.norm(magnitude)))


def main() -> None:
    neutral, neutral_f0, sample_rate = _render(MORPHOLOGY_NEUTRAL)
    neutral_spectrum = _spectrum(neutral)
    rows: list[dict[str, float | str]] = []
    for index, name in enumerate(MORPHOLOGY_CONTROLS):
        destination = 0.90
        if name in {"tract_length", "glottal_depth", "brightness"}:
            destination = 0.10
        morphology = list(MORPHOLOGY_NEUTRAL)
        morphology[index] = destination
        changed, changed_f0, _ = _render(tuple(morphology))
        changed_spectrum = _spectrum(changed)
        frequencies = np.fft.rfftfreq(len(changed), 1.0 / sample_rate)
        magnitude = np.abs(np.fft.rfft(changed * np.hanning(len(changed))))
        rows.append(
            {
                "control": name,
                "destination": destination,
                "mean_absolute_waveform_change": float(np.mean(np.abs(changed - neutral))),
                "normalized_spectrum_distance": float(
                    np.linalg.norm(changed_spectrum - neutral_spectrum)
                ),
                "spectral_centroid_hz": float(
                    np.sum(frequencies * magnitude) / max(1e-12, float(magnitude.sum()))
                ),
                "f0_delta_hz": changed_f0 - neutral_f0,
            }
        )

    destination = Path("artifacts/vocal_expression_range.csv")
    with destination.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print(destination.resolve())
    for row in rows:
        print(
            f"{row['control']}: spectrum_distance="
            f"{row['normalized_spectrum_distance']:.6f}, "
            f"f0_delta_hz={row['f0_delta_hz']:.9f}"
        )


if __name__ == "__main__":
    main()
