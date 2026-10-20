
from .base import BaseScheduler
from config import MAX_SHARED_JOBS


class SlOFirstScheduler(BaseScheduler):
    """
    Baseline:
    Prioritize the smallest SLO slack.
    Ignore security policy when choosing placements.
    Still enforce GPU memory and sharing capacity.
    """

    name = "slo_first"

    def choose_mode(self, workload, gpu):
        return "SH"

    def effective_mode(self, workload, gpu):
        return "SH"

    def resource_feasible(self, workload, gpu, mode):
        if workload.memory_gb > gpu.free_memory():
            return False

        if len(gpu.workloads) >= MAX_SHARED_JOBS:
            return False

        return True

    def choose_assignment(self, waiting_queue, gpus, current_time):
        candidates = []

        for workload in waiting_queue:
            for gpu in gpus:
                candidate = self.make_candidate(
                    workload, gpu, current_time
                )

                if candidate is not None:
                    candidates.append(candidate)

        if not candidates:
            return None

        return min(
            candidates,
            key=lambda c: (
                c["slack"],
                c["workload"].arrival_time,
            ),
        )
