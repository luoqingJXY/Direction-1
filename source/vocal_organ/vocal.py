"""A stateful vocal organ driven by continuous neural activity.

The organ receives no text, words, lyrics, entity identity, or desired waveform.
Vowel, consonant, and morphology activities are comparable to an artificial
body's available articulatory abilities: every active channel contributes to
the result, while the organ owns the acoustic implementation and continuity.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from pathlib import Path
from typing import Callable, Protocol
import wave

import numpy as np


VOWEL_UNITS = ("a", "i", "u", "e", "o", "y")
CONSONANT_UNITS = (
    "m",
    "n",
    "p",
    "b",
    "t",
    "d",
    "k",
    "g",
    "f",
    "s",
    "sh",
    "h",
    "r",
    "y",
    "w",
)
MORPHOLOGY_CONTROLS = (
    "tract_length",
    "glottal_depth",
    "brightness",
    "breathiness",
    "roughness",
    "nasality",
    "resonance",
)
MORPHOLOGY_NEUTRAL = (0.5, 0.5, 0.5, 0.0, 0.0, 0.0, 0.5)


def _bounded(value: float, name: str) -> float:
    number = float(value)
    if not math.isfinite(number) or not 0.0 <= number <= 1.0:
        raise ValueError(f"{name} must be finite and in [0, 1]")
    return number


def _activity_tuple(values: tuple[float, ...], size: int, name: str) -> tuple[float, ...]:
    result = tuple(_bounded(value, name) for value in values)
    if len(result) != size:
        raise ValueError(f"{name} must contain {size} activities, got {len(result)}")
    return result


@dataclass(frozen=True, slots=True)
class VocalControlSignal:
    """Continuous activity presented to the vocal organ by future neural tissue."""

    tick: int
    gate: float
    vowel_activity: tuple[float, ...]
    consonant_activity: tuple[float, ...]
    pitch: float
    energy: float
    airflow: float
    tension: float
    morphology_activity: tuple[float, ...] = MORPHOLOGY_NEUTRAL

    def __post_init__(self) -> None:
        object.__setattr__(self, "gate", _bounded(self.gate, "gate"))
        object.__setattr__(self, "pitch", _bounded(self.pitch, "pitch"))
        object.__setattr__(self, "energy", _bounded(self.energy, "energy"))
        object.__setattr__(self, "airflow", _bounded(self.airflow, "airflow"))
        object.__setattr__(self, "tension", _bounded(self.tension, "tension"))
        object.__setattr__(
            self,
            "vowel_activity",
            _activity_tuple(self.vowel_activity, len(VOWEL_UNITS), "vowel_activity"),
        )
        object.__setattr__(
            self,
            "consonant_activity",
            _activity_tuple(
                self.consonant_activity,
                len(CONSONANT_UNITS),
                "consonant_activity",
            ),
        )
        object.__setattr__(
            self,
            "morphology_activity",
            _activity_tuple(
                self.morphology_activity,
                len(MORPHOLOGY_CONTROLS),
                "morphology_activity",
            ),
        )

    @property
    def activities(self) -> tuple[float, ...]:
        return (
            self.gate,
            self.pitch,
            self.energy,
            self.airflow,
            self.tension,
            *self.vowel_activity,
            *self.consonant_activity,
            *self.morphology_activity,
        )

    @classmethod
    def silence(cls, tick: int) -> "VocalControlSignal":
        return cls(
            tick=tick,
            gate=0.0,
            vowel_activity=(0.0,) * len(VOWEL_UNITS),
            consonant_activity=(0.0,) * len(CONSONANT_UNITS),
            pitch=0.5,
            energy=0.0,
            airflow=0.0,
            tension=0.5,
            morphology_activity=MORPHOLOGY_NEUTRAL,
        )


@dataclass(frozen=True, slots=True)
class VocalFeedbackSignal:
    """Body-state and acoustic evidence returned by the organ."""

    tick: int
    organ_activity: tuple[float, ...]
    spectrum_activity: tuple[float, ...]
    spectrum_change: tuple[float, ...]
    rms: float
    peak: float
    f0_hz: float

    @property
    def activities(self) -> tuple[float, ...]:
        return self.organ_activity + self.spectrum_activity + self.spectrum_change


@dataclass(frozen=True, slots=True)
class VocalAudioFrame:
    tick: int
    sample_rate: int
    samples: np.ndarray
    feedback: VocalFeedbackSignal


@dataclass(frozen=True, slots=True)
class VocalRenderState:
    f0_hz: float
    envelope_start: float
    envelope_end: float
    energy: float
    airflow: float
    tension: float
    vowel_activity: tuple[float, ...]
    consonant_activity: tuple[float, ...]
    consonant_onsets: tuple[float, ...]
    morphology_activity: tuple[float, ...]


class VocalTimbreBackend(Protocol):
    """Replaceable acoustic body; it does not decide what the brain expresses."""

    sample_rate: int
    name: str

    def reset(self) -> None: ...

    def render(self, state: VocalRenderState, sample_count: int) -> np.ndarray: ...


class BrightVirtualSopranoBackend:
    """Original bright synthetic soprano timbre, built without a copied voicebank."""

    name = "bright_virtual_soprano_original"

    _FORMANTS = np.asarray(
        [
            (800.0, 1150.0, 2900.0),  # a
            (300.0, 2300.0, 3000.0),  # i
            (350.0, 900.0, 2200.0),   # u
            (500.0, 1900.0, 2600.0),  # e
            (500.0, 800.0, 2600.0),   # o
            (300.0, 1700.0, 2350.0),  # y / ü-like region
        ],
        dtype=np.float64,
    )
    _BANDWIDTHS = (90.0, 130.0, 190.0)
    _FORMANT_GAINS = (1.0, 0.58, 0.34)

    def __init__(self, sample_rate: int = 48_000, seed: int = 20260807) -> None:
        if sample_rate < 16_000:
            raise ValueError("sample_rate must be at least 16000")
        self.sample_rate = int(sample_rate)
        self._seed = int(seed)
        self.reset()

    def reset(self) -> None:
        self._phase = 0.0
        self._sample_clock = 0
        self._filter_state: dict[int, tuple[float, float]] = {}
        self._previous_noise = 0.0
        self._dc_previous_input = 0.0
        self._dc_previous_output = 0.0
        self._output_gain = 1.0
        self._rng = np.random.default_rng(self._seed)

    def _resonator(
        self,
        samples: np.ndarray,
        frequency: float,
        bandwidth: float,
        state_key: int,
    ) -> np.ndarray:
        frequency = min(max(40.0, frequency), self.sample_rate * 0.46)
        radius = math.exp(-math.pi * bandwidth / self.sample_rate)
        coefficient_1 = 2.0 * radius * math.cos(2.0 * math.pi * frequency / self.sample_rate)
        coefficient_2 = -(radius * radius)
        input_gain = 1.0 - radius
        previous_1, previous_2 = self._filter_state.get(state_key, (0.0, 0.0))
        output = np.empty_like(samples, dtype=np.float64)
        for index, sample in enumerate(samples):
            value = input_gain * float(sample) + coefficient_1 * previous_1 + coefficient_2 * previous_2
            output[index] = value
            previous_2, previous_1 = previous_1, value
        self._filter_state[state_key] = (previous_1, previous_2)
        return output

    def render(self, state: VocalRenderState, sample_count: int) -> np.ndarray:
        if sample_count <= 0:
            raise ValueError("sample_count must be positive")
        morphology = {
            name: state.morphology_activity[index]
            for index, name in enumerate(MORPHOLOGY_CONTROLS)
        }
        indices = np.arange(sample_count, dtype=np.float64)
        clock = self._sample_clock + indices
        vibrato_depth = 0.001 + 0.009 * morphology["roughness"]
        vibrato = 1.0 + vibrato_depth * np.sin(
            2.0 * math.pi * 5.2 * clock / self.sample_rate
        )
        phase_step = 2.0 * math.pi * state.f0_hz / self.sample_rate * vibrato
        phase = self._phase + np.cumsum(phase_step)
        self._phase = float(phase[-1] % (2.0 * math.pi))
        self._sample_clock += sample_count

        harmonic_count = 12 + round(8 * state.tension)
        source = np.zeros(sample_count, dtype=np.float64)
        spectral_tilt = (
            1.32
            - 0.34 * state.tension
            + 0.62 * (0.5 - morphology["glottal_depth"])
            - 0.30 * (morphology["brightness"] - 0.5)
        )
        spectral_tilt = min(2.1, max(0.55, spectral_tilt))
        for harmonic in range(1, harmonic_count + 1):
            source += np.sin(harmonic * phase) / (harmonic**spectral_tilt)
        source /= max(1.0, math.sqrt(harmonic_count))

        envelope = np.linspace(
            state.envelope_start,
            state.envelope_end,
            sample_count,
            endpoint=True,
            dtype=np.float64,
        )
        source *= envelope * (0.75 + 0.35 * state.tension)

        vowels = np.asarray(state.vowel_activity, dtype=np.float64)
        vowel_total = float(vowels.sum())
        if vowel_total <= 1e-9:
            vowels = np.asarray((0.34, 0.10, 0.14, 0.16, 0.18, 0.08), dtype=np.float64)
            vowel_total = float(vowels.sum())
        tract_scale = 1.0 + 0.34 * (0.5 - morphology["tract_length"])
        formants = vowels @ self._FORMANTS / vowel_total * tract_scale
        resonance_scale = 1.55 - 1.10 * morphology["resonance"]
        brightness_gain = 2.0 ** (1.25 * (morphology["brightness"] - 0.5))
        voiced = np.zeros(sample_count, dtype=np.float64)
        for index, (formant, bandwidth, gain) in enumerate(
            zip(formants, self._BANDWIDTHS, self._FORMANT_GAINS)
        ):
            voiced += (
                gain
                * (brightness_gain**index)
                * self._resonator(
                    source,
                    float(formant),
                    bandwidth * resonance_scale,
                    index,
                )
            )
        voiced *= 3.1

        consonants = np.asarray(state.consonant_activity, dtype=np.float64)
        onsets = np.asarray(state.consonant_onsets, dtype=np.float64)
        unit = {name: consonants[index] for index, name in enumerate(CONSONANT_UNITS)}
        onset = {name: onsets[index] for index, name in enumerate(CONSONANT_UNITS)}
        unvoiced = min(1.0, unit["p"] + unit["t"] + unit["k"] + unit["f"] + unit["s"] + unit["sh"] + unit["h"])
        nasal = min(
            1.0,
            unit["m"] + unit["n"] + 0.72 * morphology["nasality"],
        )
        stop_onset = min(
            1.0,
            onset["p"] + onset["b"] + onset["t"] + onset["d"] + onset["k"] + onset["g"],
        )
        friction = min(1.0, unit["f"] + unit["s"] + unit["sh"])

        noise = self._rng.standard_normal(sample_count)
        high_noise = np.empty_like(noise)
        high_noise[0] = noise[0] - self._previous_noise
        high_noise[1:] = noise[1:] - noise[:-1]
        self._previous_noise = float(noise[-1])
        fricative_center_numerator = (
            4200.0 * unit["f"] + 6700.0 * unit["s"] + 3300.0 * unit["sh"]
        )
        fricative_center = fricative_center_numerator / max(1e-9, friction) if friction else 4200.0
        fricative = self._resonator(high_noise, fricative_center, 900.0, 3)
        burst_envelope = np.exp(-indices / max(1.0, 0.009 * self.sample_rate))
        burst = noise * burst_envelope * stop_onset
        nasal_voice = self._resonator(source, 255.0, 95.0, 4)

        voiced *= 1.0 - 0.68 * unvoiced
        voiced = voiced * (1.0 - 0.30 * nasal) + nasal_voice * 2.1 * nasal
        rough_amplitude = 0.13 * morphology["roughness"]
        rough_modulation = 1.0 + rough_amplitude * np.sin(
            2.0 * math.pi * 27.0 * clock / self.sample_rate
        )
        voiced *= rough_modulation
        breath = noise * envelope * (
            0.018
            + 0.16 * state.airflow
            + 0.22 * morphology["breathiness"]
            + 0.16 * unit["h"]
        )
        breath += high_noise * envelope * 0.018 * morphology["roughness"]
        consonant_noise = fricative * friction * 1.8 + burst * 0.55
        signal = voiced + breath + consonant_noise

        # A stateful DC blocker and body-level output protection keep sustained
        # resonances from becoming a clipped electronic square wave.  This is an
        # organ constraint, not a neural judgement about the produced sound.
        dc_blocked = np.empty_like(signal)
        previous_input = self._dc_previous_input
        previous_output = self._dc_previous_output
        for index, sample in enumerate(signal):
            value = float(sample) - previous_input + 0.995 * previous_output
            dc_blocked[index] = value
            previous_input = float(sample)
            previous_output = value
        self._dc_previous_input = previous_input
        self._dc_previous_output = previous_output
        block_rms = math.sqrt(float(np.mean(dc_blocked * dc_blocked)))
        target_gain = min(1.0, 0.18 / max(1e-9, block_rms))
        gain_alpha = 0.82 if target_gain < self._output_gain else 0.06
        self._output_gain += gain_alpha * (target_gain - self._output_gain)
        applied_gain = min(self._output_gain, target_gain * 1.20)
        return (0.82 * np.tanh(dc_blocked * applied_gain)).astype(np.float32)


class RecordedVowelBackend:
    """A continuously controlled organ body made from isolated vowel recordings.

    The recordings contribute timbre only.  Runtime input remains the same
    numeric organ activity and never contains text or a requested utterance.
    """

    name = "recorded_vowel_material"

    def __init__(
        self,
        materials: dict[str, str | Path],
        sample_rate: int = 48_000,
        base_f0_hz: float = 270.0,
        seed: int = 20260808,
    ) -> None:
        missing = set(VOWEL_UNITS) - set(materials)
        if missing:
            raise ValueError(f"missing recorded vowel materials: {sorted(missing)}")
        self.sample_rate = int(sample_rate)
        self.base_f0_hz = float(base_f0_hz)
        self._seed = int(seed)
        self._loops = {
            name: self._load_and_prepare(Path(materials[name])) for name in VOWEL_UNITS
        }
        self.reset()

    @classmethod
    def from_directory(
        cls,
        directory: str | Path,
        sample_rate: int = 48_000,
        base_f0_hz: float = 270.0,
    ) -> "RecordedVowelBackend":
        root = Path(directory)
        return cls(
            {name: root / f"{name}.wav" for name in VOWEL_UNITS},
            sample_rate=sample_rate,
            base_f0_hz=base_f0_hz,
        )

    def _load_and_prepare(self, path: Path) -> np.ndarray:
        if not path.exists():
            raise FileNotFoundError(path)
        with wave.open(str(path), "rb") as handle:
            channels = handle.getnchannels()
            width = handle.getsampwidth()
            source_rate = handle.getframerate()
            raw = handle.readframes(handle.getnframes())
        if width != 2:
            raise ValueError(f"{path} must contain 16-bit PCM")
        samples = np.frombuffer(raw, dtype="<i2").astype(np.float64) / 32768.0
        if channels > 1:
            samples = samples.reshape(-1, channels).mean(axis=1)
        if source_rate != self.sample_rate:
            source_positions = np.arange(len(samples), dtype=np.float64)
            target_count = round(len(samples) * self.sample_rate / source_rate)
            target_positions = np.linspace(0.0, len(samples) - 1.0, target_count)
            samples = np.interp(target_positions, source_positions, samples)
        samples -= float(samples.mean())

        envelope_window = max(1, round(self.sample_rate * 0.012))
        envelope = np.convolve(
            np.abs(samples),
            np.ones(envelope_window, dtype=np.float64) / envelope_window,
            mode="same",
        )
        active = envelope > max(1e-5, float(envelope.max()) * 0.14)
        changes = np.diff(np.concatenate(([False], active, [False])).astype(np.int8))
        starts = np.flatnonzero(changes == 1)
        ends = np.flatnonzero(changes == -1)
        if not len(starts):
            raise ValueError(f"{path} contains no usable voiced material")
        longest = int(np.argmax(ends - starts))
        start, end = int(starts[longest]), int(ends[longest])
        margin = min(round(self.sample_rate * 0.025), max(0, (end - start) // 8))
        start += margin
        end -= margin
        minimum = round(self.sample_rate * 0.090)
        if end - start < minimum:
            center = (start + end) // 2
            start = max(0, center - minimum // 2)
            end = min(len(samples), start + minimum)
        maximum = round(self.sample_rate * 0.520)
        if end - start > maximum:
            center = (start + end) // 2
            start = center - maximum // 2
            end = start + maximum
        material = samples[start:end].copy()
        material -= float(material.mean())

        # A recorded vowel often retains the original speaker's syllabic
        # loudness contour.  That contour belongs to the example utterance,
        # not to the reusable timbre material.  Flatten only its slow local
        # RMS envelope so a constant organ command produces a constant sound;
        # the waveform, spectrum, and phase detail remain recorded material.
        window = max(3, round(self.sample_rate * 0.030))
        left = window // 2
        right = window - 1 - left
        squared = np.pad(material * material, (left, right), mode="reflect")
        local_power = np.convolve(
            squared,
            np.ones(window, dtype=np.float64) / window,
            mode="valid",
        )
        local_rms = np.sqrt(np.maximum(local_power, 1e-10))
        local_gain = np.clip(0.16 / local_rms, 0.45, 2.50)
        material *= local_gain

        rms = math.sqrt(float(np.mean(material * material)))
        if rms <= 1e-7:
            raise ValueError(f"{path} has no usable energy")
        return np.clip(material * (0.16 / rms), -0.72, 0.72)

    def reset(self) -> None:
        self._positions = np.zeros(len(VOWEL_UNITS), dtype=np.float64)
        self._previous_noise = 0.0
        self._morph_low_state = 0.0
        self._morph_filter_state: dict[int, tuple[float, float]] = {}
        self._sample_clock = 0
        self._rng = np.random.default_rng(self._seed)

    def _lowpass(self, samples: np.ndarray, cutoff_hz: float) -> np.ndarray:
        alpha = 1.0 - math.exp(-2.0 * math.pi * cutoff_hz / self.sample_rate)
        state = self._morph_low_state
        output = np.empty_like(samples, dtype=np.float64)
        for index, sample in enumerate(samples):
            state += alpha * (float(sample) - state)
            output[index] = state
        self._morph_low_state = state
        return output

    def _morph_resonator(
        self,
        samples: np.ndarray,
        frequency: float,
        bandwidth: float,
        state_key: int,
    ) -> np.ndarray:
        frequency = min(max(40.0, frequency), self.sample_rate * 0.46)
        radius = math.exp(-math.pi * bandwidth / self.sample_rate)
        coefficient_1 = 2.0 * radius * math.cos(2.0 * math.pi * frequency / self.sample_rate)
        coefficient_2 = -(radius * radius)
        input_gain = 1.0 - radius
        previous_1, previous_2 = self._morph_filter_state.get(state_key, (0.0, 0.0))
        output = np.empty_like(samples, dtype=np.float64)
        for index, sample in enumerate(samples):
            value = (
                input_gain * float(sample)
                + coefficient_1 * previous_1
                + coefficient_2 * previous_2
            )
            output[index] = value
            previous_2, previous_1 = previous_1, value
        self._morph_filter_state[state_key] = (previous_1, previous_2)
        return output

    def _loop(self, index: int, material: np.ndarray, count: int, rate: float) -> np.ndarray:
        crossfade = min(round(self.sample_rate * 0.018), max(8, len(material) // 5))
        period = len(material) - crossfade
        positions = (self._positions[index] + np.arange(count, dtype=np.float64) * rate) % period
        main = np.interp(positions, np.arange(len(material)), material)
        mask = positions < crossfade
        if np.any(mask):
            weights = positions[mask] / crossfade
            tail_positions = period + positions[mask]
            tail = np.interp(tail_positions, np.arange(len(material)), material)
            main[mask] = tail * (1.0 - weights) + main[mask] * weights
        self._positions[index] = float((self._positions[index] + count * rate) % period)
        return main

    def render(self, state: VocalRenderState, sample_count: int) -> np.ndarray:
        morphology = {
            name: state.morphology_activity[index]
            for index, name in enumerate(MORPHOLOGY_CONTROLS)
        }
        vowels = np.asarray(state.vowel_activity, dtype=np.float64)
        total = float(vowels.sum())
        if total <= 1e-8:
            vowels = np.asarray((0.34, 0.10, 0.14, 0.16, 0.18, 0.08), dtype=np.float64)
            total = float(vowels.sum())
        weights = vowels / total
        playback_rate = min(1.55, max(0.68, state.f0_hz / self.base_f0_hz))
        voice = np.zeros(sample_count, dtype=np.float64)
        for index, name in enumerate(VOWEL_UNITS):
            if weights[index] <= 1e-8:
                continue
            voice += weights[index] * self._loop(
                index,
                self._loops[name],
                sample_count,
                playback_rate,
            )

        tract_scale = 1.0 + 0.34 * (0.5 - morphology["tract_length"])
        tract_delta = morphology["tract_length"] - 0.5
        color_cutoff = 1_650.0 * tract_scale
        low_voice = self._lowpass(voice, color_cutoff)
        high_voice = voice - low_voice
        low_gain = 2.0 ** (
            0.85 * (morphology["glottal_depth"] - 0.5) + 0.82 * tract_delta
        )
        high_gain = 2.0 ** (
            1.75 * (morphology["brightness"] - 0.5) - 0.64 * tract_delta
        )
        voice = low_voice * low_gain + high_voice * high_gain

        resonance = self._morph_resonator(
            voice,
            1_050.0 * tract_scale,
            760.0 - 560.0 * morphology["resonance"],
            0,
        )
        voice += 0.78 * (morphology["resonance"] - 0.5) * resonance
        if morphology["nasality"] > 1e-8:
            nasal_low = self._morph_resonator(
                voice,
                270.0 * tract_scale,
                100.0,
                1,
            )
            nasal_high = self._morph_resonator(
                voice,
                1_050.0 * tract_scale,
                220.0,
                2,
            )
            nasal = morphology["nasality"]
            voice = voice * (1.0 - 0.24 * nasal) + (
                1.75 * nasal_low + 0.62 * nasal_high
            ) * nasal

        indices = self._sample_clock + np.arange(sample_count, dtype=np.float64)
        self._sample_clock += sample_count
        if morphology["roughness"] > 1e-8:
            voice *= 1.0 + 0.16 * morphology["roughness"] * np.sin(
                2.0 * math.pi * 27.0 * indices / self.sample_rate
            )

        envelope = np.linspace(
            state.envelope_start,
            state.envelope_end,
            sample_count,
            endpoint=True,
        )
        consonants = np.asarray(state.consonant_activity, dtype=np.float64)
        onsets = np.asarray(state.consonant_onsets, dtype=np.float64)
        unit = {name: consonants[index] for index, name in enumerate(CONSONANT_UNITS)}
        onset = {name: onsets[index] for index, name in enumerate(CONSONANT_UNITS)}
        unvoiced = min(1.0, unit["p"] + unit["t"] + unit["k"] + unit["f"] + unit["s"] + unit["sh"] + unit["h"])
        friction = min(1.0, unit["f"] + unit["s"] + unit["sh"])
        stop_onset = min(1.0, sum(onset[name] for name in ("p", "b", "t", "d", "k", "g")))
        noise = self._rng.standard_normal(sample_count)
        colored = np.empty_like(noise)
        colored[0] = noise[0] - self._previous_noise
        colored[1:] = noise[1:] - noise[:-1]
        self._previous_noise = float(noise[-1])
        burst = np.exp(-np.arange(sample_count) / max(1.0, self.sample_rate * 0.007))

        voice *= envelope * (1.0 - 0.48 * unvoiced)
        voice += colored * envelope * friction * 0.018
        voice += noise * envelope * (
            state.airflow * (0.010 + 0.020 * unit["h"])
            + 0.075 * morphology["breathiness"]
        )
        voice += colored * envelope * 0.014 * morphology["roughness"]
        voice += noise * burst * stop_onset * 0.045
        return (0.92 * np.tanh(voice * 1.35)).astype(np.float32)


class OneShotRecordedVowelBackend(RecordedVowelBackend):
    """Recorded timbre material that is consumed once and never looped."""

    name = "one_shot_recorded_vowel_material"

    def _loop(self, index: int, material: np.ndarray, count: int, rate: float) -> np.ndarray:
        positions = self._positions[index] + np.arange(count, dtype=np.float64) * rate
        output = np.zeros(count, dtype=np.float64)
        valid = positions < len(material) - 1
        if np.any(valid):
            output[valid] = np.interp(
                positions[valid],
                np.arange(len(material), dtype=np.float64),
                material,
            )
            fade_samples = max(1.0, self.sample_rate * 0.035 * rate)
            remaining = (len(material) - 1) - positions[valid]
            fade = np.clip(remaining / fade_samples, 0.0, 1.0)
            output[valid] *= 0.5 - 0.5 * np.cos(np.pi * fade)
        self._positions[index] = min(
            float(len(material)),
            float(self._positions[index] + count * rate),
        )
        return output

    def render(self, state: VocalRenderState, sample_count: int) -> np.ndarray:
        # Silent conditioning frames settle the organ without silently
        # consuming the single acoustic event before its gate opens.
        if state.envelope_start <= 1e-9 and state.envelope_end <= 1e-9:
            return np.zeros(sample_count, dtype=np.float32)
        start_positions = self._positions.copy()
        samples = super().render(state, sample_count)

        vowels = np.asarray(state.vowel_activity, dtype=np.float64)
        total = float(vowels.sum())
        if total <= 1e-8:
            vowels = np.asarray((0.34, 0.10, 0.14, 0.16, 0.18, 0.08), dtype=np.float64)
            total = float(vowels.sum())
        weights = vowels / total
        rate = min(1.55, max(0.68, state.f0_hz / self.base_f0_hz))
        offsets = np.arange(sample_count, dtype=np.float64) * rate
        availability = np.zeros(sample_count, dtype=np.float64)
        fade_samples = max(1.0, self.sample_rate * 0.035 * rate)
        for index, weight in enumerate(weights):
            if weight <= 1e-8:
                continue
            remaining = (len(self._loops[VOWEL_UNITS[index]]) - 1) - (
                start_positions[index] + offsets
            )
            fade = np.clip(remaining / fade_samples, 0.0, 1.0)
            availability = np.maximum(
                availability,
                0.5 - 0.5 * np.cos(np.pi * fade),
            )
        return (samples * availability).astype(np.float32)


class AuthorizedExternalBackend:
    """Adapter point for a separately authorized official or third-party engine.

    This class contains no voicebank and performs no extraction.  The supplied
    renderer is responsible for its own installation, authorization, and terms.
    """

    def __init__(
        self,
        sample_rate: int,
        renderer: Callable[[VocalRenderState, int], np.ndarray],
        name: str = "authorized_external_backend",
    ) -> None:
        self.sample_rate = int(sample_rate)
        self.renderer = renderer
        self.name = name

    def reset(self) -> None:
        reset = getattr(self.renderer, "reset", None)
        if callable(reset):
            reset()

    def render(self, state: VocalRenderState, sample_count: int) -> np.ndarray:
        samples = np.asarray(self.renderer(state, sample_count), dtype=np.float32)
        if samples.shape != (sample_count,):
            raise ValueError(f"external renderer returned {samples.shape}, expected {(sample_count,)}")
        if not np.isfinite(samples).all():
            raise ValueError("external renderer returned non-finite samples")
        return np.clip(samples, -1.0, 1.0)


class VocalOrgan:
    """Persistent artificial body organ: continuous control in, sound out."""

    def __init__(
        self,
        backend: VocalTimbreBackend | None = None,
        block_seconds: float = 0.020,
    ) -> None:
        self.backend = backend or BrightVirtualSopranoBackend()
        if not 0.002 <= block_seconds <= 0.250:
            raise ValueError("block_seconds must be in [0.002, 0.250]")
        self.block_seconds = float(block_seconds)
        self.sample_count = round(self.backend.sample_rate * self.block_seconds)
        self.reset()

    @property
    def sample_rate(self) -> int:
        return self.backend.sample_rate

    def reset(self) -> None:
        self.backend.reset()
        self._vowels = np.zeros(len(VOWEL_UNITS), dtype=np.float64)
        self._consonants = np.zeros(len(CONSONANT_UNITS), dtype=np.float64)
        self._last_consonant_target = np.zeros(len(CONSONANT_UNITS), dtype=np.float64)
        self._pitch = 0.5
        self._energy = 0.0
        self._airflow = 0.0
        self._tension = 0.5
        self._morphology = np.asarray(MORPHOLOGY_NEUTRAL, dtype=np.float64)
        self._envelope = 0.0
        self._previous_spectrum = np.zeros(8, dtype=np.float64)

    @staticmethod
    def _approach(current: np.ndarray | float, target: np.ndarray | float, alpha: float):
        return current + alpha * (target - current)

    def _spectrum(self, samples: np.ndarray) -> np.ndarray:
        windowed = samples.astype(np.float64) * np.hanning(len(samples))
        power = np.abs(np.fft.rfft(windowed)) ** 2
        frequencies = np.fft.rfftfreq(len(samples), 1.0 / self.sample_rate)
        edges = np.geomspace(70.0, self.sample_rate / 2.0, 9)
        bands = np.zeros(8, dtype=np.float64)
        for index in range(8):
            selected = power[(frequencies >= edges[index]) & (frequencies < edges[index + 1])]
            bands[index] = math.sqrt(float(selected.mean())) if selected.size else 0.0
        return np.clip(bands * (2.0 / max(1.0, len(samples))), 0.0, 1.0)

    def step(self, control: VocalControlSignal) -> VocalAudioFrame:
        duration = self.sample_count / self.sample_rate
        vowel_alpha = 1.0 - math.exp(-duration / 0.045)
        consonant_alpha = 1.0 - math.exp(-duration / 0.018)
        control_alpha = 1.0 - math.exp(-duration / 0.035)
        morphology_alpha = 1.0 - math.exp(-duration / 0.090)
        envelope_tau = 0.014 if control.gate > self._envelope else 0.065
        envelope_alpha = 1.0 - math.exp(-duration / envelope_tau)

        vowel_target = np.asarray(control.vowel_activity, dtype=np.float64)
        consonant_target = np.asarray(control.consonant_activity, dtype=np.float64)
        morphology_target = np.asarray(control.morphology_activity, dtype=np.float64)
        consonant_onsets = np.maximum(0.0, consonant_target - self._last_consonant_target)
        self._last_consonant_target = consonant_target
        self._vowels = self._approach(self._vowels, vowel_target, vowel_alpha)
        self._consonants = self._approach(self._consonants, consonant_target, consonant_alpha)
        self._pitch = float(self._approach(self._pitch, control.pitch, control_alpha))
        self._energy = float(self._approach(self._energy, control.energy, control_alpha))
        self._airflow = float(self._approach(self._airflow, control.airflow, control_alpha))
        self._tension = float(self._approach(self._tension, control.tension, control_alpha))
        self._morphology = self._approach(
            self._morphology,
            morphology_target,
            morphology_alpha,
        )
        envelope_start = self._envelope
        target_envelope = control.gate * self._energy
        self._envelope = float(self._approach(self._envelope, target_envelope, envelope_alpha))
        f0_hz = 110.0 * (2.0 ** (2.0 * self._pitch))

        render_state = VocalRenderState(
            f0_hz=f0_hz,
            envelope_start=envelope_start,
            envelope_end=self._envelope,
            energy=self._energy,
            airflow=self._airflow,
            tension=self._tension,
            vowel_activity=tuple(float(value) for value in self._vowels),
            consonant_activity=tuple(float(value) for value in self._consonants),
            consonant_onsets=tuple(float(value) for value in consonant_onsets),
            morphology_activity=tuple(float(value) for value in self._morphology),
        )
        samples = self.backend.render(render_state, self.sample_count)
        if samples.shape != (self.sample_count,) or not np.isfinite(samples).all():
            raise RuntimeError("vocal backend produced an invalid audio frame")
        samples = np.clip(samples.astype(np.float32, copy=False), -1.0, 1.0)
        spectrum = self._spectrum(samples)
        spectrum_change = np.abs(spectrum - self._previous_spectrum)
        self._previous_spectrum = spectrum
        rms = math.sqrt(float(np.mean(samples.astype(np.float64) ** 2)))
        peak = float(np.max(np.abs(samples)))
        organ_activity = (
            self._envelope,
            self._pitch,
            self._energy,
            self._airflow,
            self._tension,
            *tuple(float(value) for value in self._vowels),
            *tuple(float(value) for value in self._consonants),
            *tuple(float(value) for value in self._morphology),
        )
        feedback = VocalFeedbackSignal(
            tick=control.tick,
            organ_activity=organ_activity,
            spectrum_activity=tuple(float(value) for value in spectrum),
            spectrum_change=tuple(float(value) for value in spectrum_change),
            rms=rms,
            peak=peak,
            f0_hz=f0_hz,
        )
        return VocalAudioFrame(control.tick, self.sample_rate, samples, feedback)

    def render(self, controls: list[VocalControlSignal]) -> tuple[np.ndarray, list[VocalFeedbackSignal]]:
        frames = [self.step(control) for control in controls]
        if not frames:
            return np.zeros(0, dtype=np.float32), []
        return np.concatenate([frame.samples for frame in frames]), [frame.feedback for frame in frames]


def write_pcm16_wav(path: str | Path, samples: np.ndarray, sample_rate: int) -> None:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    bounded = np.clip(np.asarray(samples, dtype=np.float64), -1.0, 1.0)
    pcm = np.round(bounded * 32767.0).astype("<i2")
    with wave.open(str(destination), "wb") as handle:
        handle.setnchannels(1)
        handle.setsampwidth(2)
        handle.setframerate(int(sample_rate))
        handle.writeframes(pcm.tobytes())
