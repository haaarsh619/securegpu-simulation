from .base import BaseScheduler


class UnrestrictedScheduler(BaseScheduler):
    """
    Baseline:
    unrestricted ordinary sharing.
    Security policy is ignored.
    """
    name = "unrestricted"

    def choose_mode(
        self,
        workload,
        gpu,
    ) -> str:
        return "SH"

    def effective_mode(
        self,
        workload,
        gpu,
    ) -> str:
        return "SH"

    def resource_feasible(
        self,
        workload,
        gpu,
        mode,
    ) -> bool:
        if workload.memory_gb > gpu.free_memory():
            return False

        from config import MAX_SHARED_JOBS

        if len(gpu.workloads) >= MAX_SHARED_JOBS:
            return False

        return True

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
                c["gpu"].used_memory(),
            ),
        )
