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

    # Extract autoscale events
    autos_added = [s.get("autoscale_events", {}).get("added", 0) for s in stats_list]
    autos_removed = [
        s.get("autoscale_events", {}).get("removed", 0) for s in stats_list
    ]

    # Create main comparison plot (3x4 grid) with increased spacing
    fig, axes = plt.subplots(3, 4, figsize=(18, 14))
    axes = axes.flatten()

    plot_pairs = [
        (assigned, "Keys assigned"),
        (num_servers, "Number of servers"),
        (avg_loads, "Average load"),
        (stddevs, "Stddev of loads"),
        (variances, "Variance of loads"),
        (max_loads, "Max load"),
        (p95, "P95 load"),
        (percent_full, "Percent full"),
        (jain, "Jain fairness index"),
        (imbalance, "Imbalance ratio"),
        (avg_attempts, "Avg RJ attempts"),
        (autos_added, "Servers added"),
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
    plt.tight_layout(rect=[0, 0.03, 1, 0.95], w_pad=3.0, h_pad=4.0)
    summary_path = f"{out_prefix}_summary.png"
    fig.savefig(summary_path, dpi=150, bbox_inches="tight")
    plt.show()

    # Create load distribution boxplot
    loads_list = [s.get("loads", []) for s in stats_list]
    if any(loads_list):  # Only create if we have load data
        fig2, ax2 = plt.subplots(1, 1, figsize=(12, 7))
        box_plot = ax2.boxplot(loads_list, labels=labels, patch_artist=True)

        # Color the boxes
        for patch, color in zip(box_plot["boxes"], colors):
            patch.set_facecolor(color)
            patch.set_alpha(0.7)

        ax2.set_title("Per-server Load Distribution", fontsize=16, pad=20)
        ax2.set_ylabel("Load per server", fontsize=12)
        ax2.grid(axis="y", linestyle=":", alpha=0.6)

        for tick in ax2.get_xticklabels():
            tick.set_rotation(25)
            tick.set_fontsize(11)

        plt.tight_layout(pad=2.0)
        loads_path = f"{out_prefix}_loads_distribution.png"
        fig2.savefig(loads_path, dpi=150, bbox_inches="tight")
        plt.show()
    else:
        loads_path = None

    # Create autoscale events stacked bar chart
    if any(autos_added) or any(autos_removed):
        fig3, ax3 = plt.subplots(1, 1, figsize=(10, 6))
        x = range(len(labels))

        bars1 = ax3.bar(
            x, autos_added, label="Servers added", color="#2E8B57", alpha=0.8
        )
        bars2 = ax3.bar(
            x,
            autos_removed,
            bottom=autos_added,
            label="Servers removed",
            color="#DC143C",
            alpha=0.8,
        )

        ax3.set_xticks(x)
        ax3.set_xticklabels(labels, rotation=25, fontsize=11)
        ax3.set_title("Autoscaling Events", fontsize=16, pad=20)
        ax3.set_ylabel("Number of events", fontsize=12)
        ax3.legend(fontsize=11)
        ax3.grid(axis="y", linestyle=":", alpha=0.6)

        # Add value labels
        for i, (add, rem) in enumerate(zip(autos_added, autos_removed)):
            if add > 0:
                ax3.text(
                    i, add / 2, str(add), ha="center", va="center", fontweight="bold"
                )
            if rem > 0:
                ax3.text(
                    i,
                    add + rem / 2,
                    str(rem),
                    ha="center",
                    va="center",
                    fontweight="bold",
                )

        plt.tight_layout(pad=2.0)
        autoscale_path = f"{out_prefix}_autoscale.png"
        fig3.savefig(autoscale_path, dpi=150, bbox_inches="tight")
        plt.show()
    else:
        autoscale_path = None

    print(f"Saved plots with prefix '{out_prefix}':")
    print(f"  - Summary: {summary_path}")
    if loads_path:
        print(f"  - Load distribution: {loads_path}")
    if autoscale_path:
        print(f"  - Autoscale events: {autoscale_path}")

    return {
        "summary": summary_path,
        "loads_distribution": loads_path,
        "autoscale": autoscale_path,
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
