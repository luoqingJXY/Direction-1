"""人工出生结构中的稀疏物质受体记录格式。"""

from __future__ import annotations

import numpy


NEURON_MODULATION_RESPONSE_DTYPE = numpy.dtype(
    [
        ("neuron", "<u4"),
        ("material", "<u4"),
        ("gain_response", "<f4"),
        ("threshold_response", "<f4"),
    ]
)

PATH_MODULATION_RESPONSE_DTYPE = numpy.dtype(
    [
        ("path", "<u4"),
        ("material", "<u4"),
        ("response", "<f4"),
    ]
)
