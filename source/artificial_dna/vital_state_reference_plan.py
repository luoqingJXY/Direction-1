"""读取已经编译好的生命状态视觉固定出生结构数据。"""

from __future__ import annotations

import json
from pathlib import Path

from .birth_structure import VisualChannel, VisualReceptorAddress
from .vital_state_instinct import (
    PositiveEvidenceGroup,
    VitalRow,
    VitalSlotEvidence,
    VitalStateInstinctPlan,
    VitalVisualBranch,
)


COMPILED_FORMAT = 1
DEFAULT_COMPILED_PATH = (
    Path(__file__).resolve().parents[1]
    / "reference"
    / "hud_final_activities"
    / "vital_state_center_0_5.json"
)


def _group(data: dict[str, object]) -> PositiveEvidenceGroup:
    receptors = tuple(
        VisualReceptorAddress(int(value[0]), int(value[1]), VisualChannel(int(value[2])))
        for value in data["receptors"]
    )
    return PositiveEvidenceGroup(receptors, float(data["threshold"]))


def load_compiled_vital_state_plan(
    path: Path = DEFAULT_COMPILED_PATH,
) -> VitalStateInstinctPlan:
    data = json.loads(path.read_text(encoding="utf-8"))
    if int(data.get("format", -1)) != COMPILED_FORMAT:
        raise ValueError("生命状态视觉出生结构数据版本不受支持")
    if data.get("visual_size") != [1280, 657]:
        raise ValueError("生命状态视觉出生结构不是为1280×657感受场形成的")
    if data.get("visual_center") != [0.5, 0.5]:
        raise ValueError("当前出生结构只对应已经实测的中央视野位置")

    slots = tuple(
        VitalSlotEvidence(
            VitalRow(int(slot["row"])),
            int(slot["slot"]),
            tuple(_group(group) for group in slot["presence"]),
            tuple(_group(group) for group in slot["complete"]),
            tuple(_group(group) for group in slot["missing"]),
        )
        for slot in data["slots"]
    )
    return VitalStateInstinctPlan((VitalVisualBranch(0, slots),))

