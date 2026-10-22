import os

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Patch

from config import (
    TRUST_DOMAINS,
    SECURITY_LEVELS,
    SH_THRESHOLD,
    ISO_THRESHOLD,
    GPU_MEMORY_GB,
)
from generator import generate_workloads
from schedulers import get_scheduler
from security import COMPATIBILITY
from simulator import simulate


FIGURE_DIR = "figures"

# Fixed order: colour, marker and label always follow the scheduler,
# never its rank, so every figure reads the same way.
SCHEDULER_ORDER = [
    "securegpu",
    "unrestricted",
    "strict_isolation",
    "slo_first",
]

SCHEDULER_LABELS = {
    "securegpu": "SecureGPU",
    "unrestricted": "Unrestricted sharing",
    "strict_isolation": "Strict isolation",
    "slo_first": "SLO-first",
}

SCHEDULER_COLORS = {
    "securegpu": "#2a78d6",
    "unrestricted": "#eb6834",
    "strict_isolation": "#1baf7a",
    "slo_first": "#eda100",
}

# Label offsets on the trade-off scatter: the two security-blind
# baselines land close together, so push their labels apart.
TRADEOFF_LABEL_OFFSETS = {
    "securegpu": ((-10, 6), "right"),
    "strict_isolation": ((-10, -4), "right"),
    "unrestricted": ((0, -20), "center"),
    "slo_first": ((-8, 10), "right"),
}

SCHEDULER_MARKERS = {
    "securegpu": "o",
    "unrestricted": "s",
    "strict_isolation": "^",
    "slo_first": "D",
}

# SH -> ISO -> EX is an ordered scale of protection: one blue ramp.
MODE_ORDER = ["SH", "ISO", "EX"]

MODE_COLORS = {
    "SH": "#86b6ef",
    "ISO": "#2a78d6",
    "EX": "#104281",
}

MODE_LABELS = {
    "SH": "SH (ordinary sharing)",
    "ISO": "ISO (hardware-isolated)",
    "EX": "EX (exclusive)",
}

WORKLOAD_TYPES = ["inference", "fine_tuning", "training"]

WORKLOAD_LABELS = {
    "inference": "Inference",
    "fine_tuning": "Fine-tuning",
    "training": "Training",
}

INK_PRIMARY = "#0b0b0b"
INK_SECONDARY = "#52514e"
INK_MUTED = "#898781"
GRID = "#e1e0d9"
BASELINE = "#c3c2b7"
SURFACE = "#fcfcfb"
UNSAFE = "#d03b3b"


plt.rcParams.update(
    {
        "figure.facecolor": SURFACE,
        "axes.facecolor": SURFACE,
        "savefig.facecolor": SURFACE,
        "axes.edgecolor": BASELINE,
        "axes.labelcolor": INK_SECONDARY,
        "axes.titlecolor": INK_PRIMARY,
        "axes.titlesize": 11,
        "axes.labelsize": 10,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.grid": True,
        "axes.axisbelow": True,
        "grid.color": GRID,
        "grid.linewidth": 0.8,
        "xtick.color": INK_MUTED,
        "ytick.color": INK_MUTED,
        "xtick.labelcolor": INK_SECONDARY,
        "ytick.labelcolor": INK_SECONDARY,
        "legend.frameon": False,
        "legend.fontsize": 9,
        "lines.linewidth": 2,
        "lines.markersize": 6,
        "font.size": 10,
    }
)


def _path(filename):
    os.makedirs(FIGURE_DIR, exist_ok=True)
    return os.path.join(FIGURE_DIR, filename)


def _save(fig, filename):
    fig.tight_layout()
    fig.savefig(_path(filename), dpi=200)
    plt.close(fig)


def _schedulers_in(df):
    present = set(df["scheduler"].unique())
    return [s for s in SCHEDULER_ORDER if s in present]


def _mean_ci(df, group_cols, metric):
    """
    Mean and 95% confidence half-width across seeds.
    """
    grouped = df.groupby(group_cols)[metric]
    summary = grouped.agg(["mean", "std", "count"]).reset_index()
    summary["ci"] = (
        1.96
        * summary["std"].fillna(0.0)
        / np.sqrt(summary["count"])
    )
    return summary


def _scheduler_legend(fig, schedulers, ncol=4):
    handles = [
        plt.Line2D(
            [0],
            [0],
            color=SCHEDULER_COLORS[s],
            marker=SCHEDULER_MARKERS[s],
            label=SCHEDULER_LABELS[s],
        )
        for s in schedulers
    ]
    fig.legend(
        handles=handles,
        loc="lower center",
        ncol=ncol,
        bbox_to_anchor=(0.5, -0.01),
    )


# ---------------------------------------------------------------------
# Metric vs. arrival rate (one line per scheduler, 95% CI band)
# ---------------------------------------------------------------------

def _draw_metric_vs_rate(ax, df, metric, scale):
    summary = _mean_ci(df, ["scheduler", "arrival_rate"], metric)

    for scheduler in _schedulers_in(df):
        subset = summary[summary["scheduler"] == scheduler]
        x = subset["arrival_rate"]
        y = subset["mean"] * scale
        ci = subset["ci"] * scale

        ax.fill_between(
            x,
            y - ci,
            y + ci,
            color=SCHEDULER_COLORS[scheduler],
            alpha=0.15,
            linewidth=0,
        )
        ax.plot(
            x,
            y,
            color=SCHEDULER_COLORS[scheduler],
            marker=SCHEDULER_MARKERS[scheduler],
            label=SCHEDULER_LABELS[scheduler],
        )

    ax.set_xlabel("Workload arrival rate (jobs/sec)")


def plot_metric(
    df,
    metric,
    ylabel,
    title,
    filename,
    gpu_count,
    percent=False,
):
    plot_df = df[df["gpu_count"] == gpu_count]
    scale = 100 if percent else 1

    fig, ax = plt.subplots(figsize=(7, 4.5))
    _draw_metric_vs_rate(ax, plot_df, metric, scale)

    ax.set_ylabel(ylabel)
    ax.set_title(title, loc="left")

    if percent:
        ax.set_ylim(-2, 102)

    ax.legend()
    _save(fig, filename)


def plot_summary_grid(df, gpu_count):
    """
    The four metrics the evaluation is judged on, side by side.
    """
    plot_df = df[df["gpu_count"] == gpu_count]

    panels = [
        ("slo_satisfaction", "SLO satisfaction (%)"),
        ("admission_rate", "Admission rate (%)"),
        ("memory_utilization", "GPU memory utilization (%)"),
        ("security_policy_compliance", "Security-policy compliance (%)"),
    ]

    fig, axes = plt.subplots(2, 2, figsize=(11, 7.5), sharex=True)

    for ax, (metric, label) in zip(axes.flat, panels):
        _draw_metric_vs_rate(ax, plot_df, metric, 100)
        ax.set_title(label, loc="left")
        ax.set_ylim(-2, 102)

    for ax in axes[0]:
        ax.set_xlabel("")

    fig.suptitle(
        f"Scheduler comparison - {gpu_count} GPUs "
        f"(mean of seeds, shaded = 95% CI)",
        x=0.01,
        ha="left",
        color=INK_PRIMARY,
    )

    fig.tight_layout(rect=(0, 0.05, 1, 1))
    _scheduler_legend(fig, _schedulers_in(plot_df))
    fig.savefig(_path(f"summary_{gpu_count}gpus.png"), dpi=200)
    plt.close(fig)


# ---------------------------------------------------------------------
# Security vs. performance trade-off
# ---------------------------------------------------------------------

def plot_tradeoff(df):
    gpu_counts = sorted(df["gpu_count"].unique())

    fig, axes = plt.subplots(
        1,
        len(gpu_counts),
        figsize=(4.2 * len(gpu_counts), 4.4),
        sharey=True,
    )
    axes = np.atleast_1d(axes)

    for ax, gpu_count in zip(axes, gpu_counts):
        plot_df = df[df["gpu_count"] == gpu_count]

        slo = _mean_ci(plot_df, ["scheduler"], "slo_satisfaction")
        comp = _mean_ci(
            plot_df,
            ["scheduler"],
            "security_policy_compliance",
        )

        for scheduler in _schedulers_in(plot_df):
            s = slo[slo["scheduler"] == scheduler].iloc[0]
            c = comp[comp["scheduler"] == scheduler].iloc[0]

            ax.errorbar(
                c["mean"] * 100,
                s["mean"] * 100,
                xerr=c["ci"] * 100,
                yerr=s["ci"] * 100,
                color=SCHEDULER_COLORS[scheduler],
                marker=SCHEDULER_MARKERS[scheduler],
                markersize=10,
                markeredgecolor=SURFACE,
                markeredgewidth=2,
                capsize=3,
                linewidth=1,
            )
            offset, align = TRADEOFF_LABEL_OFFSETS[scheduler]
            ax.annotate(
                SCHEDULER_LABELS[scheduler],
                (c["mean"] * 100, s["mean"] * 100),
                xytext=offset,
                textcoords="offset points",
                ha=align,
                fontsize=8,
                color=INK_SECONDARY,
            )

        ax.set_title(f"{gpu_count} GPUs", loc="left")
        ax.set_xlabel("Security-policy compliance (%)")
        ax.set_xlim(-5, 110)
        ax.set_ylim(-5, 105)

    axes[0].set_ylabel("SLO satisfaction (%)")

    fig.suptitle(
        "Security vs. performance trade-off "
        "(averaged over all arrival rates; top-right is best)",
        x=0.01,
        ha="left",
        color=INK_PRIMARY,
    )

    fig.tight_layout(rect=(0, 0.06, 1, 1))
    _scheduler_legend(fig, _schedulers_in(df))
    fig.savefig(_path("tradeoff_slo_vs_compliance.png"), dpi=200)
    plt.close(fig)


# ---------------------------------------------------------------------
# Isolation-mode mix
# ---------------------------------------------------------------------

def plot_mode_mix(df):
    schedulers = _schedulers_in(df)
    totals = df.groupby("scheduler")[MODE_ORDER].sum()
    shares = totals.div(totals.sum(axis=1), axis=0) * 100

    fig, ax = plt.subplots(figsize=(8, 3.6))
    y = np.arange(len(schedulers))[::-1]
    left = np.zeros(len(schedulers))

    for mode in MODE_ORDER:
        values = shares.loc[schedulers, mode].to_numpy()

        ax.barh(
            y,
            values,
            left=left,
            color=MODE_COLORS[mode],
            edgecolor=SURFACE,
            linewidth=2,
            height=0.6,
            label=MODE_LABELS[mode],
        )

        for yi, li, vi in zip(y, left, values):
            if vi >= 8:
                ax.text(
                    li + vi / 2,
                    yi,
                    f"{vi:.0f}%",
                    ha="center",
                    va="center",
                    fontsize=8,
                    color="white" if mode != "SH" else INK_PRIMARY,
                )

        left += values

    ax.set_yticks(y)
    ax.set_yticklabels([SCHEDULER_LABELS[s] for s in schedulers])
    ax.set_xlim(0, 100)
    ax.set_xlabel("Share of placed jobs (%)")
    ax.set_title(
        "Isolation mode chosen per job (all runs)",
        loc="left",
    )
    ax.grid(axis="y", visible=False)
    ax.legend(
        loc="upper center",
        bbox_to_anchor=(0.5, -0.22),
        ncol=3,
    )

    _save(fig, "mode_mix.png")


# ---------------------------------------------------------------------
# Per-workload-type breakdown
# ---------------------------------------------------------------------

def plot_per_workload_type(df, metric_prefix, ylabel, title, filename, scale):
    gpu_counts = sorted(df["gpu_count"].unique())
    schedulers = _schedulers_in(df)

    fig, axes = plt.subplots(
        1,
        len(gpu_counts),
        figsize=(4.4 * len(gpu_counts), 4.2),
        sharey=True,
    )
    axes = np.atleast_1d(axes)

    width = 0.8 / len(schedulers)
    x = np.arange(len(WORKLOAD_TYPES))

    for ax, gpu_count in zip(axes, gpu_counts):
        plot_df = df[df["gpu_count"] == gpu_count]

        for i, scheduler in enumerate(schedulers):
            subset = plot_df[plot_df["scheduler"] == scheduler]
            means = []
            cis = []

            for workload_type in WORKLOAD_TYPES:
                values = subset[f"{metric_prefix}_{workload_type}"].dropna()
                means.append(values.mean() * scale)
                cis.append(
                    1.96 * values.std() / np.sqrt(len(values)) * scale
                    if len(values) > 1
                    else 0.0
                )

            ax.bar(
                x + (i - (len(schedulers) - 1) / 2) * width,
                means,
                width=width,
                yerr=cis,
                color=SCHEDULER_COLORS[scheduler],
                edgecolor=SURFACE,
                linewidth=2,
                error_kw={"ecolor": INK_MUTED, "elinewidth": 1, "capsize": 2},
            )

        ax.set_xticks(x)
        ax.set_xticklabels([WORKLOAD_LABELS[t] for t in WORKLOAD_TYPES])
        ax.set_title(f"{gpu_count} GPUs", loc="left")
        ax.grid(axis="x", visible=False)

    axes[0].set_ylabel(ylabel)

    fig.suptitle(
        f"{title} (averaged over all arrival rates)",
        x=0.01,
        ha="left",
        color=INK_PRIMARY,
    )

    fig.tight_layout(rect=(0, 0.06, 1, 1))
    _scheduler_legend(fig, schedulers)
    fig.savefig(_path(filename), dpi=200)
    plt.close(fig)


# ---------------------------------------------------------------------
# Unsafe co-location (grouped bars)
# ---------------------------------------------------------------------

def plot_unsafe_bars(df):
    gpu_counts = sorted(df["gpu_count"].unique())
    schedulers = _schedulers_in(df)
    summary = _mean_ci(
        df,
        ["scheduler", "gpu_count"],
        "unsafe_colocation_rate",
    )

    fig, ax = plt.subplots(figsize=(7.5, 4.2))
    width = 0.8 / len(schedulers)
    x = np.arange(len(gpu_counts))

    for i, scheduler in enumerate(schedulers):
        subset = (
            summary[summary["scheduler"] == scheduler]
            .set_index("gpu_count")
            .loc[gpu_counts]
        )
        values = subset["mean"].to_numpy() * 100
        positions = x + (i - (len(schedulers) - 1) / 2) * width

        ax.bar(
            positions,
            values,
            width=width,
            yerr=subset["ci"].to_numpy() * 100,
            color=SCHEDULER_COLORS[scheduler],
            edgecolor=SURFACE,
            linewidth=2,
            label=SCHEDULER_LABELS[scheduler],
            error_kw={"ecolor": INK_MUTED, "elinewidth": 1, "capsize": 2},
        )

        # A 0% bar is invisible; say it explicitly.
        for pos, value in zip(positions, values):
            if value < 0.5:
                ax.text(
                    pos,
                    1,
                    "0%",
                    ha="center",
                    va="bottom",
                    fontsize=8,
                    color=INK_SECONDARY,
                )

    ax.set_xticks(x)
    ax.set_xticklabels([f"{g} GPUs" for g in gpu_counts])
    ax.set_ylabel("Unsafe co-location (% of jobs)")
    ax.set_ylim(0, 100)
    ax.set_title(
        "Jobs placed with weaker protection than policy requires",
        loc="left",
    )
    ax.grid(axis="x", visible=False)
    ax.legend(loc="upper right")

    _save(fig, "unsafe_colocation.png")


# ---------------------------------------------------------------------
# Scaling with GPU count
# ---------------------------------------------------------------------

def plot_gpu_scaling(df, arrival_rates=None):
    available = sorted(df["arrival_rate"].unique())

    if arrival_rates is None:
        # Low, middle and high load from the swept range.
        arrival_rates = [
            available[0],
            available[len(available) // 2],
            available[-1],
        ]

    arrival_rates = [r for r in arrival_rates if r in available]
    schedulers = _schedulers_in(df)

    fig, axes = plt.subplots(
        1,
        len(arrival_rates),
        figsize=(4.2 * len(arrival_rates), 4.2),
        sharey=True,
    )
    axes = np.atleast_1d(axes)

    for ax, rate in zip(axes, arrival_rates):
        plot_df = df[df["arrival_rate"] == rate]
        summary = _mean_ci(
            plot_df,
            ["scheduler", "gpu_count"],
            "slo_satisfaction",
        )

        for scheduler in schedulers:
            subset = summary[summary["scheduler"] == scheduler]
            ax.errorbar(
                subset["gpu_count"],
                subset["mean"] * 100,
                yerr=subset["ci"] * 100,
                color=SCHEDULER_COLORS[scheduler],
                marker=SCHEDULER_MARKERS[scheduler],
                capsize=3,
            )

        ax.set_title(f"Arrival rate {rate:g} jobs/sec", loc="left")
        ax.set_xlabel("Number of GPUs")
        ax.set_xticks(sorted(df["gpu_count"].unique()))
        ax.set_ylim(-2, 102)

    axes[0].set_ylabel("SLO satisfaction (%)")

    fig.suptitle(
        "How many GPUs each policy needs",
        x=0.01,
        ha="left",
        color=INK_PRIMARY,
    )

    fig.tight_layout(rect=(0, 0.06, 1, 1))
    _scheduler_legend(fig, schedulers)
    fig.savefig(_path("gpu_scaling.png"), dpi=200)
    plt.close(fig)


# ---------------------------------------------------------------------
# Algorithm 1: pairwise risk -> required mode
# ---------------------------------------------------------------------

def _required_mode(risk, pair_isolation):
    if risk <= SH_THRESHOLD and pair_isolation == 1:
        return "SH"
    if risk <= ISO_THRESHOLD and pair_isolation <= 2:
        return "ISO"
    return "EX"


def plot_risk_heatmap():
    """
    rho_ij = (1 - c) * max(v_i, v_j), mapped to SH / ISO / EX
    using the configured thresholds, for each relationship
    between trust domains.
    """
    # One representative pair per compatibility value.
    relationships = {}
    for a in TRUST_DOMAINS:
        for b in TRUST_DOMAINS:
            c = COMPATIBILITY[a][b]
            relationships.setdefault(c, (a, b))

    compat_values = sorted(relationships, reverse=True)
    row_labels = []
    for c in compat_values:
        a, b = relationships[c]
        if c == 1.0:
            name = "Same domain"
        elif c >= 0.5:
            name = "Trusted partners"
        else:
            name = "Unrelated domains"
        row_labels.append(f"{name}\n(e.g. {a}-{b}, c={c:.2f})")

    levels = [1, 2, 3]
    level_names = {
        1: "Pair needs isolation level 1 (SH allowed)",
        2: "Pair needs isolation level 2",
        3: "Pair needs isolation level 3",
    }

    fig, axes = plt.subplots(
        1,
        len(levels),
        figsize=(14, 3.8),
        sharey=True,
    )

    mode_index = {m: i for i, m in enumerate(MODE_ORDER)}
    cmap = matplotlib.colors.ListedColormap(
        [MODE_COLORS[m] for m in MODE_ORDER]
    )

    for ax, level in zip(axes, levels):
        grid = np.zeros((len(compat_values), len(SECURITY_LEVELS)))

        for r, c in enumerate(compat_values):
            for k, v in enumerate(SECURITY_LEVELS):
                risk = (1.0 - c) * v
                mode = _required_mode(risk, level)
                grid[r, k] = mode_index[mode]

                ax.text(
                    k,
                    r,
                    f"{mode}\nrisk {risk:.2f}",
                    ha="center",
                    va="center",
                    fontsize=8,
                    color=INK_PRIMARY if mode == "SH" else "white",
                )

        ax.imshow(grid, cmap=cmap, vmin=-0.5, vmax=2.5, aspect="auto")

        # Thin surface gap between cells.
        ax.set_xticks(np.arange(-0.5, len(SECURITY_LEVELS)), minor=True)
        ax.set_yticks(np.arange(-0.5, len(compat_values)), minor=True)
        ax.grid(which="minor", color=SURFACE, linewidth=2)
        ax.grid(which="major", visible=False)
        ax.tick_params(which="minor", length=0)
        for spine in ax.spines.values():
            spine.set_visible(False)

        ax.set_xticks(range(len(SECURITY_LEVELS)))
        ax.set_xticklabels([f"{v:.2f}" for v in SECURITY_LEVELS])
        ax.set_xlabel("max sensitivity  max(v_i, v_j)")
        ax.set_yticks(range(len(compat_values)))
        ax.set_yticklabels(row_labels, fontsize=8)
        ax.set_title(level_names[level], loc="left", fontsize=9)

    fig.suptitle(
        "Algorithm 1: required mode for a workload pair  "
        f"(rho = (1 - c) * max(v);  SH if rho <= {SH_THRESHOLD}, "
        f"ISO if rho <= {ISO_THRESHOLD}, else EX)",
        x=0.01,
        ha="left",
        color=INK_PRIMARY,
    )

    handles = [
        Patch(facecolor=MODE_COLORS[m], label=MODE_LABELS[m])
        for m in MODE_ORDER
    ]
    fig.tight_layout(rect=(0, 0.08, 1, 1))
    fig.legend(
        handles=handles,
        loc="lower center",
        ncol=3,
        bbox_to_anchor=(0.5, -0.01),
    )
    fig.savefig(_path("algorithm1_risk_modes.png"), dpi=200)
    plt.close(fig)


# ---------------------------------------------------------------------
# Gantt timeline of one run
# ---------------------------------------------------------------------

def _assign_lanes(jobs):
    """
    Give each job on a GPU a sub-lane so co-resident jobs
    are drawn on top of each other instead of overlapping.
    """
    lane_free_at = []
    lanes = {}

    for job in sorted(jobs, key=lambda w: w.start_time):
        for lane, free_at in enumerate(lane_free_at):
            if free_at <= job.start_time + 1e-9:
                lane_free_at[lane] = job.finish_time
                lanes[job.id] = lane
                break
        else:
            lane_free_at.append(job.finish_time)
            lanes[job.id] = len(lane_free_at) - 1

    return lanes, max(1, len(lane_free_at))


def plot_gantt(
    gpu_count=6,
    arrival_rate=0.2,
    seed=1,
    num_jobs=80,
    schedulers=("securegpu", "unrestricted"),
):
    workloads = generate_workloads(
        num_jobs=num_jobs,
        arrival_rate=arrival_rate,
        seed=seed,
    )

    fig, axes = plt.subplots(
        len(schedulers),
        1,
        figsize=(12, 2.2 + 0.9 * gpu_count * len(schedulers)),
        sharex=True,
    )
    axes = np.atleast_1d(axes)

    for ax, scheduler_name in zip(axes, schedulers):
        _, jobs = simulate(
            scheduler=get_scheduler(scheduler_name),
            workloads=workloads,
            gpu_count=gpu_count,
            return_jobs=True,
        )

        yticks = []

        for gpu_id in range(gpu_count):
            gpu_jobs = [w for w in jobs if w.assigned_gpu == gpu_id]
            lanes, lane_count = _assign_lanes(gpu_jobs)
            lane_height = 0.8 / lane_count
            base = gpu_id

            for job in gpu_jobs:
                y = base - 0.4 + lanes[job.id] * lane_height

                ax.broken_barh(
                    [(job.start_time, job.finish_time - job.start_time)],
                    (y, lane_height),
                    facecolors=MODE_COLORS[job.assigned_mode],
                    edgecolors=UNSAFE if job.unsafe else SURFACE,
                    linewidth=1.2 if job.unsafe else 0.8,
                    hatch="////" if job.unsafe else None,
                )

            yticks.append(base)

        ax.set_yticks(yticks)
        ax.set_yticklabels([f"GPU {g}" for g in range(gpu_count)])
        ax.set_ylim(gpu_count - 0.5, -0.5)
        ax.grid(axis="y", visible=False)

        unsafe = sum(1 for w in jobs if w.unsafe)
        ax.set_title(
            f"{SCHEDULER_LABELS[scheduler_name]} - "
            f"{unsafe} of {len(jobs)} jobs unsafely co-located",
            loc="left",
        )

    axes[-1].set_xlabel("Simulation time (sec)")

    fig.suptitle(
        f"Placement timeline: {gpu_count} x {GPU_MEMORY_GB} GB GPUs, "
        f"{num_jobs} jobs, rate {arrival_rate:g} jobs/sec, seed {seed}  "
        "(stacked bars = co-resident jobs)",
        x=0.01,
        ha="left",
        color=INK_PRIMARY,
    )

    handles = [
        Patch(facecolor=MODE_COLORS[m], label=MODE_LABELS[m])
        for m in MODE_ORDER
    ]
    handles.append(
        Patch(
            facecolor=MODE_COLORS["SH"],
            edgecolor=UNSAFE,
            hatch="////",
            label="Unsafe co-location",
        )
    )

    fig.tight_layout(rect=(0, 0.05, 1, 0.97))
    fig.legend(
        handles=handles,
        loc="lower center",
        ncol=4,
        bbox_to_anchor=(0.5, -0.005),
    )
    fig.savefig(_path("gantt_timeline.png"), dpi=200)
    plt.close(fig)


# ---------------------------------------------------------------------

def create_all_plots(df):
    for gpu_count in sorted(df["gpu_count"].unique()):
        plot_summary_grid(df, gpu_count)

        plot_metric(
            df,
            metric="slo_satisfaction",
            ylabel="SLO satisfaction (%)",
            title=f"SLO satisfaction - {gpu_count} GPUs",
            filename=f"slo_satisfaction_{gpu_count}gpus.png",
            gpu_count=gpu_count,
            percent=True,
        )

        plot_metric(
            df,
            metric="admission_rate",
            ylabel="Admission rate (%)",
            title=f"Jobs started before their deadline - {gpu_count} GPUs",
            filename=f"admission_rate_{gpu_count}gpus.png",
            gpu_count=gpu_count,
            percent=True,
        )

        plot_metric(
            df,
            metric="memory_utilization",
            ylabel="GPU memory utilization (%)",
            title=f"GPU memory utilization - {gpu_count} GPUs",
            filename=f"memory_utilization_{gpu_count}gpus.png",
            gpu_count=gpu_count,
            percent=True,
        )

        plot_metric(
            df,
            metric="gpu_busy_fraction",
            ylabel="GPU busy time (%)",
            title=f"GPU busy time (any job resident) - {gpu_count} GPUs",
            filename=f"gpu_busy_time_{gpu_count}gpus.png",
            gpu_count=gpu_count,
            percent=True,
        )

        plot_metric(
            df,
            metric="avg_queue_delay",
            ylabel="Average queue delay (sec)",
            title=f"Average queue delay - {gpu_count} GPUs",
            filename=f"queue_delay_{gpu_count}gpus.png",
            gpu_count=gpu_count,
        )

    plot_tradeoff(df)
    plot_mode_mix(df)
    plot_unsafe_bars(df)
    plot_gpu_scaling(df)

    plot_per_workload_type(
        df,
        metric_prefix="slo",
        ylabel="SLO satisfaction (%)",
        title="SLO satisfaction by workload type",
        filename="slo_by_workload_type.png",
        scale=100,
    )

    plot_per_workload_type(
        df,
        metric_prefix="queue_delay",
        ylabel="Average queue delay (sec)",
        title="Queue delay by workload type",
        filename="queue_delay_by_workload_type.png",
        scale=1,
    )

    plot_risk_heatmap()
    plot_gantt()
