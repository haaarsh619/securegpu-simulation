from .unrestricted import UnrestrictedScheduler
from .strict_isolation import StrictIsolationScheduler
from .slo_first import SlOFirstScheduler
from .securegpu import SecureGPUScheduler


def get_scheduler(name: str):
    schedulers = {
        "unrestricted": UnrestrictedScheduler,
        "strict_isolation": StrictIsolationScheduler,
        "slo_first": SlOFirstScheduler,
        "securegpu": SecureGPUScheduler,
    }

    if name not in schedulers:
        raise ValueError(
            f"Unknown scheduler: {name}"
        )

    return schedulers[name]()
