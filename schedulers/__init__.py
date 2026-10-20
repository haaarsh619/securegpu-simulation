from .unrestricted import UnrestrictedScheduler
from .strict_isolation import StrictIsolationScheduler
from .security_only import SecurityOnlyScheduler
from .securegpu import SecureGPUScheduler


def get_scheduler(name: str):
    schedulers = {
        "unrestricted": UnrestrictedScheduler,
        "strict_isolation": StrictIsolationScheduler,
        "security_only": SecurityOnlyScheduler,
        "securegpu": SecureGPUScheduler,
    }

    if name not in schedulers:
        raise ValueError(
            f"Unknown scheduler: {name}"
        )

    return schedulers[name]()
