import random
import numpy as np

from config import (
    TRUST_DOMAINS,
    SECURITY_LEVELS,
    ISOLATION_LEVELS,
    ISOLATION_PROBS,
)
from workload import Workload


def generate_workloads(
    num_jobs=200,
    arrival_rate=0.3,
    seed=1,
):
    """
    Generate a reproducible simulated workload trace.

    These are controlled simulation inputs.
    They are not real hardware measurements.
    """
    random.seed(seed)
    np.random.seed(seed)

    workloads = []
    current_time = 0.0

    for i in range(num_jobs):
        current_time += random.expovariate(
            arrival_rate
        )

        workload_type = random.choice(
            [
                "image",
                "transformer",
                "training",
            ]
        )

        if workload_type == "image":
            memory_gb = random.randint(2, 4)
            service_ex = random.uniform(5, 15)
            service_iso = service_ex * random.uniform(1.03, 1.12)
            service_sh = service_ex * random.uniform(1.10, 1.30)

        elif workload_type == "transformer":
            memory_gb = random.randint(6, 10)
            service_ex = random.uniform(10, 30)
            service_iso = service_ex * random.uniform(1.05, 1.18)
            service_sh = service_ex * random.uniform(1.20, 1.45)

        else:
            memory_gb = random.randint(8, 12)
            service_ex = random.uniform(25, 60)
            service_iso = service_ex * random.uniform(1.04, 1.15)
            service_sh = service_ex * random.uniform(1.10, 1.30)

        trust_domain = random.choice(
            TRUST_DOMAINS
        )

        sensitivity = random.choice(
            SECURITY_LEVELS
        )

        min_isolation = random.choices(
            ISOLATION_LEVELS,
            weights=ISOLATION_PROBS,
            k=1,
        )[0]

        # Proposal-style relative SLO budget
        budget_multiplier = random.uniform(
            1.5,
            3.0,
        )

        deadline = (
            current_time
            + service_ex * budget_multiplier
        )

        workloads.append(
            Workload(
                id=i,
                workload_type=workload_type,
                arrival_time=current_time,
                memory_gb=memory_gb,
                trust_domain=trust_domain,
                sensitivity=sensitivity,
                min_isolation=min_isolation,
                service_sh=service_sh,
                service_iso=service_iso,
                service_ex=service_ex,
                deadline=deadline,
            )
        )

    return workloads
