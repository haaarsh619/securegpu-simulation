from generator import generate_workloads
from schedulers import get_scheduler
from simulator import simulate


def main():
    workloads = generate_workloads(
        num_jobs=30,
        arrival_rate=0.3,
        seed=1,
    )

    scheduler_names = [
        "unrestricted",
        "strict_isolation",
        "security_only",
        "securegpu",
    ]

    for name in scheduler_names:
        scheduler = get_scheduler(
            name
        )

        result = simulate(
            scheduler=scheduler,
            workloads=workloads,
            gpu_count=4,
        )

        print(
            f"\n{name}"
        )

        for key, value in result.items():
            print(
                f"  {key}: {value}"
            )


if __name__ == "__main__":
    main()
