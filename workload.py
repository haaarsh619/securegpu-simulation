from dataclasses import dataclass
from typing import Optional


@dataclass
class Workload:
    id: int
    workload_type: str
    arrival_time: float
    memory_gb: int

    trust_domain: str
    sensitivity: float
    min_isolation: int

    service_sh: float
    service_iso: float
    service_ex: float

    deadline: float

    start_time: Optional[float] = None
    finish_time: Optional[float] = None
    assigned_gpu: Optional[int] = None
    assigned_mode: Optional[str] = None
    unsafe: bool = False

    def service_time(self, mode: str) -> float:
        if mode == "SH":
            return self.service_sh
        if mode == "ISO":
            return self.service_iso
        return self.service_ex

    def queue_delay(self) -> float:
        if self.start_time is None:
            return 0.0
        return self.start_time - self.arrival_time

    def completion_time(self) -> float:
        if self.finish_time is None:
            return 0.0
        return self.finish_time - self.arrival_time

    def met_slo(self) -> bool:
        return (
            self.finish_time is not None
            and self.finish_time <= self.deadline
        )
