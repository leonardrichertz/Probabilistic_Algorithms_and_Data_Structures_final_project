from time import time
from config import (
    NUM_KEYS,
    NUM_SERVERS,
    SERVER_CAPACITY,
    K,
)
from data_generator import generate_uniform_keys, generate_zipfian_keys
from hashing_algorithms import (
    BoundedHashRing_CH_BL,
    BoundedHashRing_RJ_CH,
    BoundedHashRing_RehashThreshold,
)
import statistics

from metrics import (
    calculate_distribution_metrics,
    measure_lookup_time,
    measure_memory_overhead,
)
import csv


# The jain index is a common metric for measuring load balance. It measures how evenly the load is distributed across servers.
# A jain index of 1 indicates perfect balance (all servers have the same load), while lower values indicate more imbalance.
def _jain_index(loads):
    if not loads:
        return 0.0
    s = sum(loads)
    s2 = sum(l * l for l in loads)
    n = len(loads)
    return (s * s) / (n * s2) if s2 > 0 else 0.0


def _detailed_stats_from_ring(
    ring,
    total_keys,
    total_attempts=0,
    total_hashes=0,
    avg_insertion_time=0,
    autoscale_events=None,
):
    loads = list(ring.server_loads.values())
    n = len(loads) or 1
    total_assigned = len(ring.key_assignments)
    avg = sum(loads) / n if loads else 0
    var = statistics.pvariance(loads) if loads else 0
    std = statistics.pstdev(loads) if loads else 0
    mx = max(loads) if loads else 0
    loads_sorted = sorted(loads)
    p95 = loads_sorted[int(0.95 * len(loads_sorted))] if loads else 0
    percent_full = sum(1 for l in loads if l >= ring.capacity) / n
    jain = _jain_index(loads)
    imbalance = mx / avg if avg > 0 else float("inf")
    avg_attempts = total_attempts / max(1, total_assigned)
    avg_hashes = total_hashes / max(1, total_assigned)

    return {
        "assigned": total_assigned,
        "total_keys": total_keys,
        "num_servers": n,
        "avg_load": avg,
        "stddev": std,
        "variance": var,
        "max_load": mx,
        "p95_load": p95,
        "percent_full": percent_full,
        "jain_index": jain,
        "imbalance_ratio": imbalance,
        "avg_attempts": avg_attempts,
        "avg_hashes": avg_hashes,
        "avg_insertion_time": avg_insertion_time,
        "autoscale_events": autoscale_events or {"added": 0, "removed": 0},
        "loads": loads,
        "ring": ring,
    }


# Fixed-k CH (first clockwise)
def run_fixed_k_ch(keys):
    ring = BoundedHashRing_CH_BL(capacity=SERVER_CAPACITY)
    for i in range(NUM_SERVERS):
        ring.add_server(f"S{i}")
    total_attempts = 0
    total_hashes = 0
    total_insertion_time = 0

    for k in keys:
        start_time = time()
        _, cost, hashes = ring.assign_key(k)
        end_time = time()
        total_attempts += cost
        total_hashes += hashes
        total_insertion_time += end_time - start_time

    avg_insertion_time = total_insertion_time / len(keys) if keys else 0

    return _detailed_stats_from_ring(
        ring,
        len(keys),
        total_attempts,
        total_hashes,
        avg_insertion_time,
        autoscale_events=None,
    )


# Fixed-k RJ (Random-Jump)
def run_fixed_k_rj(keys):
    ring = BoundedHashRing_RJ_CH(capacity=SERVER_CAPACITY, k=K)
    for i in range(NUM_SERVERS):
        ring.add_server(f"S{i}")
    total_hashes = 0
    total_attempts = 0
    total_insertion_time = 0

    for k in keys:
        start_time = time()
        _, cost, hashes = ring.assign_key(k)
        end_time = time()
        total_attempts += cost
        total_hashes += hashes
        total_insertion_time += end_time - start_time

    avg_insertion_time = total_insertion_time / len(keys) if keys else 0

    return _detailed_stats_from_ring(
        ring, len(keys), total_attempts, total_hashes, avg_insertion_time
    )


# Dynamic-k RJ (autoscaling) -> we find the best server and if that doesn't satisfy threshold, we rehash up to k times
def run_dynamic_k_rj(keys):
    ring = BoundedHashRing_RehashThreshold(capacity=SERVER_CAPACITY, k=K)
    for i in range(NUM_SERVERS):
        ring.add_server(f"S{i}")
    total_attempts = 0
    total_hashes = 0
    total_insertion_time = 0

    for k in keys:
        start_time = time()
        _, cost, hashes = ring.assign_key(k)
        end_time = time()
        total_attempts += cost
        total_hashes += hashes
        total_insertion_time += end_time - start_time

    avg_insertion_time = total_insertion_time / len(keys) if keys else 0

    return _detailed_stats_from_ring(
        ring, len(keys), total_attempts, total_hashes, avg_insertion_time
    )


def evaluate_metrics():
    keys = generate_zipfian_keys(NUM_KEYS, total_possible_keys=1000, alpha=1.5)

    # Run simulations
    fixed_ch_stats = run_fixed_k_ch(keys)
    fixed_rj_stats = run_fixed_k_rj(keys)
    dynamic_rj_stats = run_dynamic_k_rj(keys)

    # Evaluate distribution metrics
    for stats, label in zip(
        [fixed_ch_stats, fixed_rj_stats, dynamic_rj_stats],
        ["Fixed-k CH", "Fixed-k RJ", "Dynamic-k RJ"],
    ):
        metrics = calculate_distribution_metrics(stats["loads"])
        print(f"{label} Distribution Metrics: {metrics}")

    # Evaluate computational efficiency
    lookup_time = measure_lookup_time(dynamic_rj_stats["ring"], keys)
    memory_overhead = measure_memory_overhead(dynamic_rj_stats["ring"])
    print(f"Dynamic-k RJ Lookup Time: {lookup_time:.6f} seconds")
    print(f"Dynamic-k RJ Memory Overhead: {memory_overhead} bytes")


def run_multiple_simulations():
    key_multipliers = [0.5, 0.75, 1.0, 1.5, 2.0]
    skewed_results = []
    uniform_results = []

    for multiplier in key_multipliers:
        num_keys = int(multiplier * NUM_KEYS)
        skewed_keys = generate_zipfian_keys(num_keys)
        uniform_keys = generate_uniform_keys(num_keys)

        skewed_fixed_ch_stats = run_fixed_k_ch(skewed_keys)
        skewed_fixed_rj_stats = run_fixed_k_rj(skewed_keys)
        skewed_dynamic_rj_stats = run_dynamic_k_rj(skewed_keys)

        uniform_fixed_ch_stats = run_fixed_k_ch(uniform_keys)
        uniform_fixed_rj_stats = run_fixed_k_rj(uniform_keys)
        uniform_dynamic_rj_stats = run_dynamic_k_rj(uniform_keys)

        skewed_results.append(
            {
                "multiplier": multiplier,
                "fixed_ch": skewed_fixed_ch_stats,
                "fixed_rj": skewed_fixed_rj_stats,
                "dynamic_rj": skewed_dynamic_rj_stats,
            }
        )

        uniform_results.append(
            {
                "multiplier": multiplier,
                "fixed_ch": uniform_fixed_ch_stats,
                "fixed_rj": uniform_fixed_rj_stats,
                "dynamic_rj": uniform_dynamic_rj_stats,
            }
        )

        # Generate plots for each run
        from createplots import plot_stats_comparison

        plot_stats_comparison(
            [skewed_fixed_ch_stats, skewed_fixed_rj_stats, skewed_dynamic_rj_stats],
            labels=["Skewed Fixed-k CH", "Skewed Fixed-k RJ", "Skewed Dynamic-k RJ"],
            out_prefix=f"skewed_comparison_multiplier_{multiplier}",
        )

        plot_stats_comparison(
            [uniform_fixed_ch_stats, uniform_fixed_rj_stats, uniform_dynamic_rj_stats],
            labels=["Uniform Fixed-k CH", "Uniform Fixed-k RJ", "Uniform Dynamic-k RJ"],
            out_prefix=f"uniform_comparison_multiplier_{multiplier}",
        )

    save_results_to_csv("skewed_results.csv", skewed_results)
    save_results_to_csv("uniform_results.csv", uniform_results)

    return skewed_results, uniform_results


def save_results_to_csv(filename, results):
    """Save simulation results to a CSV file."""
    with open(filename, mode="w", newline="") as file:
        writer = csv.writer(file)

        # Write the header
        writer.writerow(
            [
                "Multiplier",
                "Algorithm",
                "Assigned",
                "Avg Load",
                "Stddev",
                "Variance",
                "Max Load",
                "Jain Index",
                "Imbalance Ratio",
                "Avg Attempts",
                "Avg Hashes",
                "Avg Insertion Time",
            ]
        )

        # Write the data
        for result in results:
            multiplier = result["multiplier"]
            for algorithm, stats in result.items():
                if algorithm == "multiplier":
                    continue
                writer.writerow(
                    [
                        multiplier,
                        algorithm,
                        stats["assigned"],
                        stats["avg_load"],
                        stats["stddev"],
                        stats["variance"],
                        stats["max_load"],
                        stats["jain_index"],
                        stats["imbalance_ratio"],
                        stats["avg_attempts"],
                        stats["avg_hashes"],
                        stats["avg_insertion_time"],
                    ]
                )


# Call the function to run the simulations
if __name__ == "__main__":
    skewed_results, uniform_results = run_multiple_simulations()
    print("Skewed Results:", skewed_results)
    print("Uniform Results:", uniform_results)
    evaluate_metrics()
