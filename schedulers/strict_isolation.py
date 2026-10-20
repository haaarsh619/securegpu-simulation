from .base import BaseScheduler


class StrictIsolationScheduler(BaseScheduler):
    """
    Baseline:
    every workload receives exclusive GPU execution.
    """
    name = "strict_isolation"

    def choose_mode(
        self,
        workload,
        gpu,
    ) -> str:
        return "EX"

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
                c["gpu"].id,
            ),
        )
