import pandas as pd

from config import (
    DEFAULT_NUM_JOBS,
    DEFAULT_GPU_COUNTS,
    DEFAULT_ARRIVAL_RATES,
    DEFAULT_SEEDS,
)
from generator import generate_workloads
from schedulers import get_scheduler
from simulator import simulate
from plot_results import create_all_plots


SCHEDULER_NAMES = [
    "unrestricted",
    "strict_isolation",
    "slo_first",
    "securegpu",
]


def run_experiments():
    results = []

    for gpu_count in DEFAULT_GPU_COUNTS:
        for arrival_rate in DEFAULT_ARRIVAL_RATES:
            for seed in DEFAULT_SEEDS:

                # Same workload trace for every scheduler.
                workloads = generate_workloads(
                    num_jobs=DEFAULT_NUM_JOBS,
                    arrival_rate=arrival_rate,
                    seed=seed,
                )

                for scheduler_name in SCHEDULER_NAMES:
                    scheduler = get_scheduler(
                        scheduler_name
                    )

                    result = simulate(
                        scheduler=scheduler,
                        workloads=workloads,
                        gpu_count=gpu_count,
                    )

                    result[
                        "gpu_count"
                    ] = gpu_count

                    result[
                        "arrival_rate"
                    ] = arrival_rate

                    result[
                        "seed"
                    ] = seed

                    results.append(
                        result
                    )

                    print(
                        f"{scheduler_name:18s} "
                        f"GPUs={gpu_count:2d} "
                        f"rate={arrival_rate:.1f} "
                        f"seed={seed}"
                    )

    df = pd.DataFrame(
        results
    )

    df.to_csv(
        "securegpu_results.csv",
        index=False,
    )

    return df


def print_summary(df):
    summary_columns = [
        "slo_satisfaction",
        "unsafe_colocation_rate",
        "avg_queue_delay",
        "avg_completion_time",
        "memory_utilization",
        "gpu_busy_fraction",
    ]

    summary = (
        df
        .groupby(
            "scheduler"
        )[summary_columns]
        .mean()
        .round(4)
    )

    print(
        "\nAVERAGE RESULTS\n"
    )

    print(
        summary
    )


if __name__ == "__main__":
    df = run_experiments()

    print_summary(
        df
    )

    create_all_plots(
        df
    )

    print(
        "\nCreated:"
    )

    print(
        "securegpu_results.csv"
    )

    print(
        "slo_satisfaction.png"
    )

    print(
        "unsafe_colocation.png"
    )

    print(
        "gpu_utilization.png"
    )
