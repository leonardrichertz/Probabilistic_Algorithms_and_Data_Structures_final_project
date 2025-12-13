import matplotlib.pyplot as plt


def plot_stats_comparison(stats_list, labels=None, out_prefix="comparison"):
    """
    Create and save comparison plots for a list of stats dictionaries.

    Args:
      stats_list: list of dicts (each dict has keys listed below).
      labels: list of strings for legend (same length as stats_list). If None, generic labels are used.
      out_prefix: filename prefix for saved images.

    Expected keys in each stats dict:
      "assigned", "total_keys", "num_servers", "avg_load", "stddev", "variance",
      "max_load", "p95_load", "percent_full", "jain_index", "imbalance_ratio",
      "avg_attempts", "autoscale_events" (dict with "added" and "removed"), "loads"
    """
    if labels is None:
        labels = [f"run_{i}" for i in range(len(stats_list))]

    # Extract scalar metrics
    assigned = [s.get("assigned", 0) for s in stats_list]
    total_keys = [s.get("total_keys", 0) for s in stats_list]
    num_servers = [s.get("num_servers", 0) for s in stats_list]
    avg_loads = [s.get("avg_load", 0) for s in stats_list]
    stddevs = [s.get("stddev", 0) for s in stats_list]
    variances = [s.get("variance", 0) for s in stats_list]
    max_loads = [s.get("max_load", 0) for s in stats_list]
    p95 = [s.get("p95_load", 0) for s in stats_list]
    percent_full = [s.get("percent_full", 0) for s in stats_list]
    jain = [s.get("jain_index", 0) for s in stats_list]
    imbalance = [s.get("imbalance_ratio", 0) for s in stats_list]
    avg_attempts = [s.get("avg_attempts", 0) for s in stats_list]
    avg_hashes = [s.get("avg_hashes", 0) for s in stats_list]
    avg_insertion_times = [s.get("avg_insertion_time", 0) for s in stats_list]

    # Create main comparison plot (4x4 grid) with increased spacing
    fig, axes = plt.subplots(3, 4, figsize=(18, 18))
    axes = axes.flatten()

    # Removed autoscaling metrics from the plot pairs
    plot_pairs = [
        (assigned, "Keys assigned"),
        (avg_loads, "Average load"),
        (stddevs, "Stddev of loads"),
        (variances, "Variance of loads"),
        (max_loads, "Max load"),
        (percent_full, "Percent full"),
        (jain, "Jain fairness index"),
        (imbalance, "Imbalance ratio"),
        (avg_attempts, "Avg RJ attempts"),
        (avg_insertion_times, "Average Insertion Time"),
    ]

    colors = plt.cm.Set3.colors
    for i, (series, title) in enumerate(plot_pairs):
        ax = axes[i]
        bars = ax.bar(
            labels, series, color=[colors[j % len(colors)] for j in range(len(labels))]
        )
        ax.set_title(title, fontsize=12, pad=15)
        ax.grid(axis="y", linestyle=":", alpha=0.6)

        # Rotate x-axis labels for better readability
        for tick in ax.get_xticklabels():
            tick.set_rotation(25)
            tick.set_fontsize(9)

        # Add value labels on bars
        for bar, val in zip(bars, series):
            if val > 0:
                ax.text(
                    bar.get_x() + bar.get_width() / 2,
                    bar.get_height() + max(series) * 0.01,
                    f"{val:.2f}" if isinstance(val, float) else f"{val}",
                    ha="center",
                    va="bottom",
                    fontsize=8,
                )

    # Hide unused subplots
    for i in range(len(plot_pairs), len(axes)):
        axes[i].axis("off")

    plt.suptitle("Algorithm Performance Comparison", fontsize=18, y=0.98)
    # Increased spacing between subplots
    plt.tight_layout(rect=[0, 0.03, 1, 0.95], w_pad=5.0, h_pad=10.0)
    summary_path = f"{out_prefix}_summary.png"
    fig.savefig(summary_path, dpi=150, bbox_inches="tight")
    plt.show()

    print(f"Saved plots with prefix '{out_prefix}':")
    print(f"  - Summary: {summary_path}")

    return {
        "summary": summary_path,
    }


def plot_single_stats(
    stats_dict, title="Algorithm Performance", out_file="single_stats.png"
):
    """
    Create a summary plot for a single stats dictionary.
    """
    plot_stats_comparison(
        [stats_dict], labels=[title], out_prefix=out_file.replace(".png", "")
    )
