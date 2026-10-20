import matplotlib.pyplot as plt


def plot_metric(
    df,
    metric,
    ylabel,
    title,
    filename,
    gpu_count=8,
    percent=False,
):
    plot_df = df[
        df["gpu_count"] == gpu_count
    ]

    summary = (
        plot_df
        .groupby(
            [
                "scheduler",
                "arrival_rate",
            ]
        )
        .mean(
            numeric_only=True
        )
        .reset_index()
    )

    plt.figure(
        figsize=(8, 5)
    )

    for scheduler in summary[
        "scheduler"
    ].unique():
        subset = summary[
            summary["scheduler"]
            == scheduler
        ]

        y = subset[
            metric
        ]

        if percent:
            y = y * 100

        plt.plot(
            subset["arrival_rate"],
            y,
            marker="o",
            label=scheduler,
        )

    plt.xlabel(
        "Workload Arrival Rate (jobs/sec)"
    )

    plt.ylabel(
        ylabel
    )

    plt.title(
        title
    )

    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()

    plt.savefig(
        filename,
        dpi=200,
    )

    plt.close()


def create_all_plots(df):
    for gpu_count in [3, 6, 9]:
        plot_metric(
            df,
            metric="slo_satisfaction",
            ylabel="SLO Satisfaction (%)",
            title=f"SLO Satisfaction - {gpu_count} GPUs",
            filename=f"slo_satisfaction_{gpu_count}gpus.png",
            gpu_count=gpu_count,
            percent=True,
        )

        plot_metric(
            df,
            metric="unsafe_colocation_rate",
            ylabel="Unsafe Co-location (%)",
            title=f"Unsafe Co-location - {gpu_count} GPUs",
            filename=f"unsafe_colocation_{gpu_count}gpus.png",
            gpu_count=gpu_count,
            percent=True,
        )

        plot_metric(
            df,
            metric="gpu_busy_fraction",
            ylabel="GPU Busy Time (%)",
            title=f"GPU Utilization - {gpu_count} GPUs",
            filename=f"gpu_utilization_{gpu_count}gpus.png",
            gpu_count=gpu_count,
            percent=True,
        )