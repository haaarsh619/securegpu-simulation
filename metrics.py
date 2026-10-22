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

    # Admission = the job was placed on a GPU before its SLO
    # deadline passed (start_time <= deadline).
    admitted = [
        w for w in completed
        if w.start_time is not None
        and w.start_time <= w.deadline
    ]

    per_type = {}

    for workload_type in [
        "inference",
        "fine_tuning",
        "training",
    ]:
        jobs_of_type = [
            w for w in completed
            if w.workload_type == workload_type
        ]

        if not jobs_of_type:
            per_type[f"slo_{workload_type}"] = np.nan
            per_type[f"queue_delay_{workload_type}"] = np.nan
            continue

        per_type[f"slo_{workload_type}"] = (
            sum(1 for w in jobs_of_type if w.met_slo())
            / len(jobs_of_type)
        )

        per_type[f"queue_delay_{workload_type}"] = float(
            np.mean([w.queue_delay() for w in jobs_of_type])
        )

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
        "slo_violation_rate": (
            1.0 - len(slo_met) / len(completed)
        ),
        "admission_rate": (
            len(admitted) / len(completed)
        ),
        "unsafe_colocation_rate": (
            len(unsafe_jobs) / len(completed)
        ),
        "security_policy_compliance": (
            1.0 - len(unsafe_jobs) / len(completed)
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
        **per_type,
    }
