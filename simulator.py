import copy
import heapq

from config import GPU_MEMORY_GB
from gpu import GPU
from metrics import calculate_metrics


def simulate(
    scheduler,
    workloads,
    gpu_count=4,
    return_jobs=False,
):
    jobs = copy.deepcopy(
        workloads
    )

    gpus = [
        GPU(i)
        for i in range(gpu_count)
    ]

    waiting_queue = []
    completed = []

    event_queue = []
    event_counter = 0

    jobs_by_id = {
        w.id: w
        for w in jobs
    }

    # Create arrival events.
    for workload in jobs:
        heapq.heappush(
            event_queue,
            (
                workload.arrival_time,
                event_counter,
                "arrival",
                workload.id,
                None,
            ),
        )
        event_counter += 1

    current_time = 0.0
    last_event_time = 0.0

    memory_area = 0.0
    busy_gpu_area = 0.0

    while event_queue:
        event_time = event_queue[0][0]

        # Measure resource use between events.
        delta = event_time - last_event_time

        total_used_memory = sum(
            gpu.used_memory()
            for gpu in gpus
        )

        busy_gpus = sum(
            1
            for gpu in gpus
            if not gpu.is_empty()
        )

        memory_area += (
            total_used_memory
            * delta
        )

        busy_gpu_area += (
            busy_gpus
            * delta
        )

        current_time = event_time

        same_time_events = []

        while (
            event_queue
            and event_queue[0][0] == event_time
        ):
            same_time_events.append(
                heapq.heappop(
                    event_queue
                )
            )

        # Process completions first.
        for event in same_time_events:
            _, _, event_type, workload_id, gpu_id = event

            if event_type != "completion":
                continue

            workload = jobs_by_id[
                workload_id
            ]

            gpu = gpus[
                gpu_id
            ]

            gpu.remove_workload(
                workload
            )

            workload.finish_time = (
                current_time
            )

            completed.append(
                workload
            )

        # Process arrivals.
        for event in same_time_events:
            _, _, event_type, workload_id, _ = event

            if event_type != "arrival":
                continue

            waiting_queue.append(
                jobs_by_id[
                    workload_id
                ]
            )

        # Keep scheduling until no further feasible
        # workload-GPU pair exists at this time.
        while True:
            assignment = (
                scheduler.choose_assignment(
                    waiting_queue,
                    gpus,
                    current_time,
                )
            )

            if assignment is None:
                break

            workload = assignment[
                "workload"
            ]

            gpu = assignment[
                "gpu"
            ]

            mode = assignment[
                "mode"
            ]

            service_time = assignment[
                "service_time"
            ]

            workload.start_time = (
                current_time
            )

            workload.assigned_gpu = (
                gpu.id
            )

            workload.assigned_mode = (
                mode
            )

            workload.unsafe = assignment[
                "unsafe"
            ]

            gpu.add_workload(
                workload,
                mode,
            )

            waiting_queue.remove(
                workload
            )

            finish_time = (
                current_time
                + service_time
            )

            heapq.heappush(
                event_queue,
                (
                    finish_time,
                    event_counter,
                    "completion",
                    workload.id,
                    gpu.id,
                ),
            )

            event_counter += 1

        last_event_time = event_time

    total_time = max(
        w.finish_time
        for w in completed
        if w.finish_time is not None
    )

    result = calculate_metrics(
        completed=completed,
        gpu_count=gpu_count,
        total_time=total_time,
        memory_area=memory_area,
        busy_gpu_area=busy_gpu_area,
        gpu_memory_gb=GPU_MEMORY_GB,
    )

    result["scheduler"] = scheduler.name

    if return_jobs:
        return result, completed

    return result
