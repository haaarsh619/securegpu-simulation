import numpy as np


def calculate_metrics(
    completed,
    gpu_count,
    total_time,
    memory_area,
    busy_gpu_area,
    gpu_memory_gb,
):
    if not completed:
        return {}

    slo_met = [
        w for w in completed
        if w.met_slo()
    ]

    unsafe_jobs = [
        w for w in completed
        if w.unsafe
    ]

    queue_delays = [
        w.queue_delay()
        for w in completed
    ]

    completion_times = [
        w.completion_time()
        for w in completed
    ]

    mode_counts = {
        mode: sum(
            1
            for w in completed
            if w.assigned_mode == mode
        )
        for mode in ["SH", "ISO", "EX"]
    }

    memory_utilization = (
        memory_area
        /
        (
            total_time
            * gpu_count
            * gpu_memory_gb
        )
    )

    gpu_busy_fraction = (
        busy_gpu_area
        /
        (
            total_time
            * gpu_count
        )
    )

    return {
        "jobs": len(completed),
        "slo_satisfaction": (
            len(slo_met) / len(completed)
        ),
        "unsafe_colocation_rate": (
            len(unsafe_jobs) / len(completed)
        ),
        "avg_queue_delay": float(
            np.mean(queue_delays)
        ),
        "avg_completion_time": float(
            np.mean(completion_times)
        ),
        "memory_utilization": float(
            memory_utilization
        ),
        "gpu_busy_fraction": float(
            gpu_busy_fraction
        ),
        "SH": mode_counts["SH"],
        "ISO": mode_counts["ISO"],
        "EX": mode_counts["EX"],
    }
