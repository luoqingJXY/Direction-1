"""Commission every non-brain interface with deterministic physical fixtures."""

from __future__ import annotations

import argparse
import secrets
import time
from pathlib import Path

from artificial_brain.actuation import ActuationGate, DryRunMotorSink
from artificial_brain.body_observation import PeripheralEvidenceRecorder
from artificial_brain.body_runtime import (
    BodyPeripheralRuntime,
    NullAudioOutput,
    SyntheticAudioSource,
    SyntheticScreenSource,
    peripheral_frame_payload,
)
from artificial_brain.peripheral_signals import BodyCommand, MotorControlSignal, SafetyState, TargetState
from artificial_brain.signal_transport import AuthenticatedPacketCodec, PacketError
from artificial_brain.vocal import VocalControlSignal


def verify_safety() -> dict[str, object]:
    sink = DryRunMotorSink()
    gate = ActuationGate(sink, max_mouse_pixels_per_tick=12, command_ttl_ms=100.0)
    now = time.perf_counter_ns()
    signal = MotorControlSignal(0, now, 1.0, -1.0, (0.8,) * 8)
    exact = TargetState(True, True, True, True, 1280, 720)
    cases = {
        "unarmed": SafetyState(False, False, exact, 0.0),
        "emergency": SafetyState(True, True, exact, 0.0),
        "wrong_process": SafetyState(True, False, TargetState(True, True, False, True), 0.0),
        "not_foreground": SafetyState(True, False, TargetState(True, False, True, True), 0.0),
        "heartbeat_timeout": SafetyState(True, False, exact, 101.0),
    }
    blocked = {name: not gate.apply(signal, state, now, now).allowed for name, state in cases.items()}
    allowed = gate.apply(signal, SafetyState(True, False, exact, 0.0), now, now)
    return {
        "blocked_cases": blocked,
        "all_forbidden_cases_blocked": all(blocked.values()),
        "bounded_allowed_mouse": [allowed.mouse_dx, allowed.mouse_dy] == [12, -12],
        "release_count": sink.release_count,
    }


def verify_transport() -> dict[str, bool]:
    secret = secrets.token_bytes(32)
    sender = AuthenticatedPacketCodec(secret, "commission")
    receiver = AuthenticatedPacketCodec(secret, "commission")
    packet = sender.encode(0, {"activity": [0.2, 0.4]})
    authenticated = receiver.decode(packet)["payload"]["activity"] == [0.2, 0.4]
    try:
        receiver.decode(packet)
        replay_blocked = False
    except PacketError:
        replay_blocked = True
    damaged = bytearray(sender.encode(1, {"activity": [0.6]}))
    damaged[-1] ^= 1
    try:
        receiver.decode(bytes(damaged))
        tamper_blocked = False
    except PacketError:
        tamper_blocked = True
    return {"authenticated": authenticated, "replay_blocked": replay_blocked, "tamper_blocked": tamper_blocked}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--ticks", type=int, default=150)
    parser.add_argument("--period-ms", type=float, default=20.0)
    parser.add_argument("--output", type=Path, default=Path("artifacts/peripheral_commission"))
    args = parser.parse_args()
    if args.ticks <= 0 or args.period_ms <= 0.0:
        raise SystemExit("ticks and period must be positive")

    motor_sink = DryRunMotorSink()
    audio_output = NullAudioOutput()
    runtime = BodyPeripheralRuntime(
        SyntheticScreenSource(),
        SyntheticAudioSource(block_seconds=args.period_ms / 1000.0),
        audio_output,
        ActuationGate(motor_sink),
    )
    exact = TargetState(True, True, True, True, 320, 180)
    safety = SafetyState(True, False, exact, 0.0)
    secret = secrets.token_bytes(32)
    sender = AuthenticatedPacketCodec(secret, "body-to-brain")
    receiver = AuthenticatedPacketCodec(secret, "body-to-brain")
    recorder = PeripheralEvidenceRecorder()

    next_tick_ns = time.perf_counter_ns()
    for tick in range(args.ticks):
        now_ns = time.perf_counter_ns()
        if now_ns < next_tick_ns:
            time.sleep((next_tick_ns - now_ns) / 1_000_000_000.0)
        received_ns = time.perf_counter_ns()
        command = BodyCommand(
            tick,
            MotorControlSignal.still(tick, received_ns),
            VocalControlSignal.silence(tick),
        )
        frame = runtime.step(command, safety, command_received_ns=received_ns)
        packet = sender.encode(tick, peripheral_frame_payload(frame))
        decoded = receiver.decode(packet)
        if decoded["sequence"] != tick:
            raise RuntimeError("transport broke tick identity")
        recorder.record(frame, len(packet), args.period_ms)
        next_tick_ns += round(args.period_ms * 1_000_000.0)

    safety_result = verify_safety()
    transport_result = verify_transport()
    base = recorder.summary(args.period_ms)
    acceptance = {
        "tick_continuity": bool(base["tick_continuity"]),
        "visual_activity_changes": int(base["visual_change_ticks"]) > 0,
        "audio_activity_changes": int(base["audio_change_ticks"]) > 0,
        "one_vocal_frame_per_tick": audio_output.frames_written == args.ticks,
        "processing_p95_within_period": float(base["processing_p95_ms"]) <= args.period_ms,
        "cycle_jitter_p95_within_2ms": float(base["cycle_jitter_p95_ms"]) <= 2.0,
        "safety_fail_closed": bool(safety_result["all_forbidden_cases_blocked"]),
        "authenticated_transport": all(transport_result.values()),
    }
    summary = recorder.write(
        args.output,
        args.period_ms,
        {
            "visual_field": [32, 18],
            "audio_bands": 24,
            "motor_channels": 10,
            "vocal_channels": 33,
            "safety": safety_result,
            "transport": transport_result,
            "acceptance": acceptance,
            "all_acceptance_passed": all(acceptance.values()),
            "scope": "non-brain engineering commissioning; not training and not a life experiment",
        },
    )
    print(f"non-brain commissioning: {'PASS' if summary['all_acceptance_passed'] else 'FAIL'}")
    print(f"ticks={summary['ticks']} p95={summary['processing_p95_ms']:.3f} ms max={summary['processing_max_ms']:.3f} ms")
    print(args.output.resolve())


if __name__ == "__main__":
    main()
