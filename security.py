from config import (
    TRUST_DOMAINS,
    SH_THRESHOLD,
    ISO_THRESHOLD,
    MODE_STRENGTH,
    STRENGTH_MODE,
)
from workload import Workload
from gpu import GPU


def build_compatibility_matrix():
    """
    Policy-defined compatibility matrix.
    Same domain = 1.00
    Trusted partner pairs = 0.75
    Unrelated domains = 0.25
    """
    matrix = {}
    trusted_pairs = [
        {"A", "B"},
        {"C", "D"},
    ]

    for a in TRUST_DOMAINS:
        matrix[a] = {}

        for b in TRUST_DOMAINS:
            if a == b:
                matrix[a][b] = 1.00
            elif {a, b} in trusted_pairs:
                matrix[a][b] = 0.75
            else:
                matrix[a][b] = 0.25

    return matrix


COMPATIBILITY = build_compatibility_matrix()


def pairwise_risk(w1: Workload, w2: Workload) -> float:
    """
    Proposal equation:
    rho_ij = (1 - c_zizj) * max(v_i, v_j)
    """
    compatibility = COMPATIBILITY[
        w1.trust_domain
    ][
        w2.trust_domain
    ]

    return (
        1.0 - compatibility
    ) * max(
        w1.sensitivity,
        w2.sensitivity,
    )


def pair_required_mode(w1: Workload, w2: Workload) -> str:
    """
    Proposal rule:
      SH  if risk <= tau_s and pair isolation = 1
      ISO if risk <= tau_h and pair isolation <= 2
      EX  otherwise
    """
    risk = pairwise_risk(w1, w2)
    pair_isolation = max(
        w1.min_isolation,
        w2.min_isolation,
    )

    if (
        risk <= SH_THRESHOLD
        and pair_isolation == 1
    ):
        return "SH"

    if (
        risk <= ISO_THRESHOLD
        and pair_isolation <= 2
    ):
        return "ISO"

    return "EX"


def required_mode_for_gpu(
    workload: Workload,
    gpu: GPU,
) -> str:
    """
    Strongest required protection across:
      - workload's own minimum isolation
      - every pair with current GPU residents
    """
    required_strength = workload.min_isolation

    for resident in gpu.workloads:
        pair_mode = pair_required_mode(
            workload,
            resident,
        )

        required_strength = max(
            required_strength,
            MODE_STRENGTH[pair_mode],
        )

    return STRENGTH_MODE[
        required_strength
    ]


def max_candidate_risk(
    workload: Workload,
    gpu: GPU,
) -> float:
    if gpu.is_empty():
        return 0.0

    return max(
        pairwise_risk(workload, resident)
        for resident in gpu.workloads
    )


def unsafe_if_shared(
    workload: Workload,
    gpu: GPU,
    actual_mode: str,
) -> bool:
    """
    A placement is unsafe if the mode used is weaker
    than the required mode for any workload pair.
    """
    if gpu.is_empty():
        return False

    for resident in gpu.workloads:
        required = pair_required_mode(
            workload,
            resident,
        )

        if (
            MODE_STRENGTH[actual_mode]
            < MODE_STRENGTH[required]
        ):
            return True

    return False
