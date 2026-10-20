from dataclasses import dataclass, field
from typing import List, Optional

from config import GPU_MEMORY_GB
from workload import Workload


@dataclass
class GPU:
    id: int
    memory_total: int = GPU_MEMORY_GB
    workloads: List[Workload] = field(default_factory=list)
    mode: Optional[str] = None

    def used_memory(self) -> int:
        return sum(w.memory_gb for w in self.workloads)

    def free_memory(self) -> int:
        return self.memory_total - self.used_memory()

    def is_empty(self) -> bool:
        return len(self.workloads) == 0

    def add_workload(self, workload: Workload, mode: str) -> None:
        if self.is_empty():
            self.mode = mode
        self.workloads.append(workload)

    def remove_workload(self, workload: Workload) -> None:
        if workload in self.workloads:
            self.workloads.remove(workload)

        if self.is_empty():
            self.mode = None
