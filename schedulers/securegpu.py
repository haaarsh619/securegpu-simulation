from .base import BaseScheduler


class SecureGPUScheduler(BaseScheduler):
    """
    Proposal algorithm:
      1. Security compatibility determines required protection
      2. Resource feasibility filters candidates
      3. Smallest SLO slack wins
      4. Lower risk breaks ties
      5. Earlier arrival breaks remaining ties
    """
    name = "securegpu"

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
                c["slack"],
                c["risk"],
                c["workload"].arrival_time,
            ),
        )
