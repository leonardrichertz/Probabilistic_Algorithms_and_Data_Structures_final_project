import numpy as np


def calculate_distribution_metrics(server_loads):
    """
    Calculate distribution quality metrics for server loads.

    Args:
        server_loads (list): List of server loads.

    Returns:
        dict: Dictionary containing standard deviation, peak-to-average ratio, and min-max spread.
    """
    if not server_loads:
        return {"std_dev": 0, "peak_to_avg": 0, "min_max_spread": 0}

    mean_load = np.mean(server_loads)
    max_load = np.max(server_loads)
    min_load = np.min(server_loads)
    std_dev = np.std(server_loads)

    return {
        "std_dev": std_dev,
        "peak_to_avg": max_load / mean_load if mean_load > 0 else 0,
        "min_max_spread": max_load - min_load,
    }


def measure_lookup_time(ring, keys):
    """
    Measure the average lookup time for a set of keys.

    Args:
        ring (ConsistentHashRing): The hash ring.
        keys (list): List of keys to look up.

    Returns:
        float: Average lookup time in seconds.
    """
    import time

    start_time = time.time()
    for key in keys:
        ring.get_server_for_hash(ring._hash(key))
    end_time = time.time()
    return (end_time - start_time) / len(keys)


def measure_memory_overhead(ring):
    """
    Measure the memory overhead of the hash ring.

    Args:
        ring (ConsistentHashRing): The hash ring.

    Returns:
        int: Memory usage in bytes.
    """
    import sys

    return sys.getsizeof(ring.ring) + sys.getsizeof(ring.server_map)
