from typing import Optional, Dict

from config import (
    MAX_SHARED_JOBS,
    MODE_STRENGTH,
)
from security import (
    required_mode_for_gpu,
    max_candidate_risk,
    unsafe_if_shared,
)


class BaseScheduler:
    name = "base"

    def choose_mode(
        self,
        workload,
        gpu,
    ) -> str:
        return required_mode_for_gpu(
            workload,
            gpu,
        )

    def resource_feasible(
        self,
        workload,
        gpu,
        mode,
    ) -> bool:
        # Memory check
        if workload.memory_gb > gpu.free_memory():
            return False

        # EX requires an empty GPU
        if mode == "EX":
            return gpu.is_empty()

        # SH and ISO allow co-residency up to configured max
        if len(gpu.workloads) >= MAX_SHARED_JOBS:
            return False

        # Proposal assumption:
        # an occupied GPU keeps its current mode until residents finish.
        if not gpu.is_empty():
            if gpu.mode == "EX":
                return False

            # Cannot require a stronger mode than the occupied GPU currently uses.
            if (
                MODE_STRENGTH[mode]
                > MODE_STRENGTH[gpu.mode]
            ):
                return False

            # Candidate runs under the current GPU mode.
            mode = gpu.mode

        return True

    def effective_mode(
        self,
        workload,
        gpu,
    ) -> str:
        required_mode = self.choose_mode(
            workload,
            gpu,
        )

        if gpu.is_empty():
            return required_mode

        return gpu.mode

    def make_candidate(
        self,
        workload,
        gpu,
        current_time,
    ) -> Optional[Dict]:
        required_mode = self.choose_mode(
            workload,
            gpu,
        )

        if not self.resource_feasible(
            workload,
            gpu,
            required_mode,
        ):
            return None

        mode = self.effective_mode(
            workload,
            gpu,
        )

        service_time = workload.service_time(
            mode
        )

        slack = (
            workload.deadline
            - current_time
            - service_time
        )

        return {
            "workload": workload,
            "gpu": gpu,
            "required_mode": required_mode,
            "mode": mode,
            "service_time": service_time,
            "slack": slack,
            "risk": max_candidate_risk(
                workload,
                gpu,
            ),
            "unsafe": unsafe_if_shared(
                workload,
                gpu,
                mode,
            ),
        }

    def choose_assignment(
        self,
        waiting_queue,
        gpus,
        current_time,
    ):
        raise NotImplementedError
