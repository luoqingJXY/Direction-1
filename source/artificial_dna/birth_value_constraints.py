"""在人工DNA先确定结构目标后，由冻结统一公式推出的出生数值边界。

这里不是 Brain 组件，也不形成生命 Signal。它只供人工出生结构生成阶段核对：
当两个已经独立传播的来源在同一个神经元直接相加，并且该处需要区分
“任一来源单独到达”和“两来源共同到达”时，响应强度、路径衰减与阈值
是否存在自洽的数值区间。

本文件只推导区间，不提供默认响应强度、默认路径强度或默认阈值。
"""

from __future__ import annotations

from dataclasses import dataclass
import math


def _finite_nonnegative(value: float, name: str) -> float:
    result = float(value)
    if not math.isfinite(result) or result < 0.0:
        raise ValueError(f"{name}必须是有限非负数值")
    return result


@dataclass(frozen=True, slots=True)
class IndependentContinuationStage:
    """独立继续通道中一个来源神经元及其向后固定路径的出生值。"""

    neuron_response_gain: float
    neuron_threshold: float
    fixed_path_strength: float

    def __post_init__(self) -> None:
        gain = _finite_nonnegative(
            self.neuron_response_gain,
            "独立继续神经元活动响应强度",
        )
        _finite_nonnegative(self.neuron_threshold, "独立继续神经元阈值")
        strength = _finite_nonnegative(
            self.fixed_path_strength,
            "独立继续固定路径强度",
        )
        if gain <= 0.0:
            raise ValueError("独立继续神经元响应强度必须大于零")
        if not 0.0 < strength <= 1.0:
            raise ValueError("独立继续固定路径强度必须处于(0,1]")

    @property
    def transfer_coefficient(self) -> float:
        return float(self.neuron_response_gain) * float(self.fixed_path_strength)

    def require_every_positive_activity_to_continue(self) -> None:
        """连续非负活动没有最小正值，因此阈值只能为零。"""

        if float(self.neuron_threshold) != 0.0:
            raise ValueError(
                "独立继续通道的正阈值会截断一部分正活动，不能完整保留内部差异"
            )


@dataclass(frozen=True, slots=True)
class IndependentContinuationBounds:
    source_minimum: float
    source_maximum: float
    result_minimum: float
    result_maximum: float
    transfer_coefficient: float

    @property
    def preserves_exact_strength(self) -> bool:
        return math.isclose(
            self.transfer_coefficient,
            1.0,
            rel_tol=1e-12,
            abs_tol=1e-12,
        )


def derive_exact_contact_neuron_values(
    fixed_path_strength: float,
) -> tuple[float, float]:
    """在人工DNA选择原强度到达后，推出满足该选择的唯一神经元值。

    根据 ``Q = g × U × P``，要求每个 ``U`` 都有 ``Q = U``，必须
    ``g = 1/P``；又因为活动是连续非负强度，要让任意正活动传播，阈值
    必须为零。当前器官入口的固定路径已经确定为 ``P=1``，因此结果为
    ``g=1, 阈值=0``。
    """

    strength = _finite_nonnegative(fixed_path_strength, "客观接触固定路径强度")
    if not 0.0 < strength <= 1.0:
        raise ValueError("客观接触固定路径强度必须处于(0,1]")
    gain = 1.0 / strength
    if not math.isfinite(gain):
        raise OverflowError("客观接触神经元响应强度超过当前物理数值容纳范围")
    return gain, 0.0


def derive_independent_continuation_bounds(
    source_minimum: float,
    source_maximum: float,
    *stages: IndependentContinuationStage,
) -> IndependentContinuationBounds:
    """推导多节独立继续后的活动范围，不替任何一节选择出生值。"""

    minimum = _finite_nonnegative(source_minimum, "独立继续来源最小活动")
    maximum = _finite_nonnegative(source_maximum, "独立继续来源最大活动")
    if minimum > maximum:
        raise ValueError("独立继续来源最小活动不能大于最大活动")
    if not stages:
        raise ValueError("独立继续范围至少需要一节实际神经元和固定路径")
    coefficient = 1.0
    for stage in stages:
        stage.require_every_positive_activity_to_continue()
        coefficient *= stage.transfer_coefficient
        if not math.isfinite(coefficient):
            raise OverflowError("独立继续倍率超过当前物理数值容纳范围")
    return IndependentContinuationBounds(
        minimum,
        maximum,
        minimum * coefficient,
        maximum * coefficient,
        coefficient,
    )


@dataclass(frozen=True, slots=True)
class FixedPathArrivalBounds:
    """一条固定路径来源活动及其实际到达活动的范围。"""

    source_minimum: float
    source_maximum: float
    path_strength: float

    def __post_init__(self) -> None:
        minimum = _finite_nonnegative(self.source_minimum, "来源最小活动")
        maximum = _finite_nonnegative(self.source_maximum, "来源最大活动")
        strength = _finite_nonnegative(self.path_strength, "固定路径强度")
        if minimum > maximum:
            raise ValueError("来源最小活动不能大于来源最大活动")
        if not 0.0 < strength <= 1.0:
            raise ValueError("固定路径强度必须处于(0,1]，路径不能放大活动")

    @property
    def arrival_minimum(self) -> float:
        return float(self.source_minimum) * float(self.path_strength)

    @property
    def arrival_maximum(self) -> float:
        return float(self.source_maximum) * float(self.path_strength)


@dataclass(frozen=True, slots=True)
class ArrivalSumBounds:
    """任意数量普通到达活动直接相加后的完整范围。"""

    minimum: float
    maximum: float
    arrival_count: int

    def __post_init__(self) -> None:
        minimum = _finite_nonnegative(self.minimum, "多路到达最小总和")
        maximum = _finite_nonnegative(self.maximum, "多路到达最大总和")
        if minimum > maximum:
            raise ValueError("多路到达最小总和不能大于最大总和")
        if int(self.arrival_count) <= 0:
            raise ValueError("多路到达至少需要一条实际路径")


def derive_arrival_sum_bounds(
    *arrivals: FixedPathArrivalBounds,
) -> ArrivalSumBounds:
    """严格按 ``S=ΣQ`` 推出任意多路固定路径的直接相加范围。"""

    if not arrivals:
        raise ValueError("多路到达范围至少需要一条实际路径")
    minimum = sum(value.arrival_minimum for value in arrivals)
    maximum = sum(value.arrival_maximum for value in arrivals)
    if not math.isfinite(minimum) or not math.isfinite(maximum):
        raise OverflowError("多路到达总和超过当前物理数值容纳范围")
    return ArrivalSumBounds(minimum, maximum, len(arrivals))


@dataclass(frozen=True, slots=True)
class OutputSafeResponseGainBounds:
    """在全部到达直接相加后保持目标活动上限的响应强度区间。"""

    lower_inclusive: float
    upper_inclusive: float
    target_activity_maximum: float
    arrival_sum_maximum: float

    def __post_init__(self) -> None:
        lower = _finite_nonnegative(self.lower_inclusive, "响应强度下界")
        upper = _finite_nonnegative(self.upper_inclusive, "响应强度上界")
        target = _finite_nonnegative(
            self.target_activity_maximum,
            "目标活动最大容纳量",
        )
        arrival = _finite_nonnegative(
            self.arrival_sum_maximum,
            "到达活动最大总和",
        )
        if lower > upper:
            raise ValueError("响应强度下界不能大于上界")
        if target <= 0.0 or arrival <= 0.0:
            raise ValueError("目标活动上限和到达活动最大总和必须大于零")

    def accepts(self, response_gain: float) -> bool:
        value = _finite_nonnegative(response_gain, "神经元活动响应强度")
        return self.lower_inclusive <= value <= self.upper_inclusive


def derive_output_safe_response_gain_bounds(
    target_activity_maximum: float,
    arrivals: ArrivalSumBounds,
) -> OutputSafeResponseGainBounds:
    """按 ``A=gS`` 推出不会超过客观输出活动上限的全部 ``g``。

    这里只给出可行区间，不替人工出生结构选择区间中的具体响应强度。
    """

    target = _finite_nonnegative(
        target_activity_maximum,
        "目标活动最大容纳量",
    )
    if target <= 0.0:
        raise ValueError("目标活动最大容纳量必须大于零")
    maximum = float(arrivals.maximum)
    if maximum <= 0.0:
        raise ValueError("到达活动最大总和必须大于零")
    upper = target / maximum
    if not math.isfinite(upper):
        raise OverflowError("输出安全响应强度边界超过当前物理数值容纳范围")
    return OutputSafeResponseGainBounds(0.0, upper, target, maximum)


@dataclass(frozen=True, slots=True)
class TwoArrivalThresholdInterval:
    """拒绝任一单独来源、允许两来源共同到达的阈值区间。

    下界必须严格大于，上界可以等于；这与统一公式中的传播条件
    ``A >= 阈值`` 完全一致。
    """

    lower_exclusive: float
    upper_inclusive: float

    @property
    def feasible(self) -> bool:
        return self.lower_exclusive < self.upper_inclusive

    def accepts(self, threshold: float) -> bool:
        value = _finite_nonnegative(threshold, "神经元出生阈值")
        # 下界在理论上严格排除，不能使用近似相等放宽；上界在理论上允许
        # 相等，需要容纳十进制出生值经过浮点乘加产生的最后一位表示差异。
        at_or_below_upper = value <= self.upper_inclusive or math.isclose(
            value,
            self.upper_inclusive,
            rel_tol=1e-12,
            abs_tol=1e-12,
        )
        return self.lower_exclusive < value and at_or_below_upper

    def require_feasible(self) -> None:
        if not self.feasible:
            raise ValueError(
                "两路来源的当前活动范围没有形成可分离阈值区间；"
                "不能用任意阈值伪造共同到达关系"
            )


def derive_two_arrival_threshold_interval(
    target_response_gain: float,
    first: FixedPathArrivalBounds,
    second: FixedPathArrivalBounds,
) -> TwoArrivalThresholdInterval:
    """按 ``A=g×(Q₁+Q₂)`` 推出共同到达阈值的完整可行区间。

    若要保证任一路即使达到各自最大活动也不能单独继续传播，阈值必须
    严格大于：

    ``g × max(Q₁最大, Q₂最大)``

    若又要保证两路都达到各自指定最小活动时能够继续传播，阈值必须
    不大于：

    ``g × (Q₁最小 + Q₂最小)``
    """

    gain = _finite_nonnegative(target_response_gain, "目标神经元活动响应强度")
    lower = gain * max(first.arrival_maximum, second.arrival_maximum)
    upper = gain * (first.arrival_minimum + second.arrival_minimum)
    if not math.isfinite(lower) or not math.isfinite(upper):
        raise OverflowError("共同到达数值边界超过当前物理数值容纳范围")
    return TwoArrivalThresholdInterval(lower, upper)


def derive_all_arrivals_required_threshold_interval(
    target_response_gain: float,
    *arrivals: FixedPathArrivalBounds,
) -> TwoArrivalThresholdInterval:
    """推出“缺少任一路都拒绝、全部到达才允许”的统一阈值区间。

    所有活动非负。为了拒绝任意一路缺失而其余各路达到最大值的情况，
    阈值必须严格大于所有“缺一路”组合中的最大活动；为了允许全部来源
    仅达到各自指定最小值的情况，阈值又必须不大于全部最小到达之和。

    返回区间可能不可行，尤其当连续活动的最小正值没有被人工出生结构
    明确为大于零时。不可行就是该组连续活动不能被固定阈值冒充布尔与门。
    """

    gain = _finite_nonnegative(target_response_gain, "目标神经元活动响应强度")
    if len(arrivals) < 2:
        raise ValueError("全部到达关系至少需要两路独立来源")
    maxima = [value.arrival_maximum for value in arrivals]
    minima = [value.arrival_minimum for value in arrivals]
    total_maximum = sum(maxima)
    lower = gain * (total_maximum - min(maxima))
    upper = gain * sum(minima)
    if not math.isfinite(lower) or not math.isfinite(upper):
        raise OverflowError("全部到达阈值边界超过当前物理数值容纳范围")
    return TwoArrivalThresholdInterval(lower, upper)
