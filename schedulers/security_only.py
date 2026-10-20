from .base import BaseScheduler


class SecurityOnlyScheduler(BaseScheduler):
    """
    Security-compatible placement without SLO ordering.
    FIFO among feasible security-safe placements.
    """
    name = "security_only"

    def choose_assignment(
        self,
        waiting_queue,
        gpus,
        current_time,
    ):
        candidates = []

        for workload in waiting_queue:
            for gpu in gpus:
                candidate = self.make_candidate(
                    workload,
                    gpu,
                    current_time,
                )

                if candidate is not None:
                    candidates.append(
                        candidate
                    )

        if not candidates:
            return None

        return min(
            candidates,
            key=lambda c: (
                c["workload"].arrival_time,
                c["risk"],
                c["gpu"].id,
            ),
        )
