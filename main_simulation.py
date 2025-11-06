# main_simulation.py

import numpy as np
from hashing_algorithms import BoundedHashRing_CH_BL, BoundedHashRing_RJ_CH
from dlb_model import LearnedHashRing_DLB
from data_generator import generate_uniform_keys, generate_zipfian_keys
from config import NUM_SERVERS, SERVER_CAPACITY, NUM_KEYS_UNIFORM, NUM_KEYS_SKEWED
from collections import defaultdict


def simulate_load(hashing_ring, keys):
    """Runs a load simulation and records metrics."""
    # Reset metrics and loads
    hashing_ring.server_loads = defaultdict(int)
    hashing_ring.key_assignments = {}

    total_assigned = 0
    total_search_cost = 0

    # Ensure all specialized metrics are reset
    if hasattr(hashing_ring, "total_cascades"):
        hashing_ring.total_cascades = 0
    if hasattr(hashing_ring, "total_jumps"):
        hashing_ring.total_jumps = 0
    if hasattr(hashing_ring, "total_predictions"):
        hashing_ring.total_predictions = 0

    for key in keys:
        server, cost = hashing_ring.assign_key(key)

        if server:
            total_assigned += 1
            total_search_cost += cost
        else:
            # System overload; stop processing keys
            break

    # Calculate load variance
    loads = list(hashing_ring.server_loads.values())
    if not loads:
        return {}

    avg_load = sum(loads) / len(loads)
    variance = sum((l - avg_load) ** 2 for l in loads) / len(loads)

    return {
        "algorithm": hashing_ring.__class__.__name__,
        "total_keys": len(keys),
        "assigned_keys": total_assigned,
        "avg_search_cost": total_search_cost / max(1, total_assigned),
        "load_variance": variance,
        "full_servers": sum(1 for load in loads if load == hashing_ring.capacity),
        "server_loads": loads,
    }


def run_comparison(keys, description):
    """Initializes rings and runs the simulation."""
    servers = [f"Server-{i}" for i in range(NUM_SERVERS)]

    # Initialize all rings
    ch_bl_ring = BoundedHashRing_CH_BL(capacity=SERVER_CAPACITY)
    rj_ch_ring = BoundedHashRing_RJ_CH(capacity=SERVER_CAPACITY)

    # DLB must be initialized last as its init triggers model training
    for s in servers:
        ch_bl_ring.add_server(s)
        rj_ch_ring.add_server(s)

    # DLB Ring Initialization (Triggers Training)
    dlb_ring = LearnedHashRing_DLB(capacity=SERVER_CAPACITY)
    for s in servers:
        dlb_ring.add_server(s)

    print(f"\n--- Running {description} Simulation ---")

    results = [
        simulate_load(ch_bl_ring, keys),
        simulate_load(rj_ch_ring, keys),
        simulate_load(dlb_ring, keys),
    ]

    # Print and compare results
    for res in results:
        print(f"\n{'-'*10} {res['algorithm']} {'-'*10}")
        print(f"Total Keys Assigned: {res['assigned_keys']}/{res['total_keys']}")
        print(f"Servers Overloaded: {res['full_servers']}")
        print(f"Avg. Search/Cost: {res['avg_search_cost']:.4f}")
        print(f"Load Variance: {res['load_variance']:.2f}")

    return results


if __name__ == "__main__":

    # --- PHASE 1: UNIFORM LOAD COMPARISON (CH-BL vs RJ-CH focus) ---
    uniform_keys = generate_uniform_keys(NUM_KEYS_UNIFORM)
    run_comparison(uniform_keys, "UNIFORM LOAD")

    # --- PHASE 2: SKEWED LOAD COMPARISON (DLB focus) ---
    skewed_keys = generate_zipfian_keys(NUM_KEYS_SKEWED)
    run_comparison(skewed_keys, "SKEWED LOAD (HOT KEYS)")
