import math
from config import NUM_SERVERS, SERVER_CAPACITY, NUM_KEYS_UNIFORM, NUM_VIRTUAL_NODES
from data_generator import generate_uniform_keys
from hashing_algorithms import BoundedHashRing_CH_BL, BoundedHashRing_RJ_CH
import statistics


def _jain_index(loads):
    if not loads:
        return 0.0
    s = sum(loads)
    s2 = sum(l * l for l in loads)
    n = len(loads)
    return (s * s) / (n * s2) if s2 > 0 else 0.0


def _stats_from_ring(ring, total_keys):
    loads = list(ring.server_loads.values())
    avg = sum(loads) / len(loads) if loads else 0
    var = sum((l - avg) ** 2 for l in loads) / len(loads) if loads else 0
    assigned = len(ring.key_assignments)
    return {
        "assigned": assigned,
        "total_keys": total_keys,
        "avg_load": avg,
        "variance": var,
        "loads": loads,
    }


def _detailed_stats_from_ring(
    ring, total_keys, total_attempts=0, autoscale_events=None
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
        "autoscale_events": autoscale_events or {"added": 0, "removed": 0},
        "loads": loads,
    }


def run_fixed_k_ch(keys):
    ring = BoundedHashRing_CH_BL(capacity=SERVER_CAPACITY)
    for i in range(NUM_SERVERS):
        ring.add_server(f"S{i}")
    total_attempts = 0
    for k in keys:
        _, cost = ring.assign_key(k)
        total_attempts += cost
    return _detailed_stats_from_ring(ring, len(keys), total_attempts)


def run_fixed_k_rj(keys):
    ring = BoundedHashRing_RJ_CH(
        capacity=SERVER_CAPACITY, vnodes=NUM_VIRTUAL_NODES, max_attempts=50
    )
    for i in range(NUM_SERVERS):
        ring.add_server(f"S{i}")
    total_attempts = 0
    for k in keys:
        _, cost = ring.assign_key(k)
        total_attempts += cost
    return _detailed_stats_from_ring(ring, len(keys), total_attempts)


def run_dynamic_k_rj(
    keys,
    min_servers=None,
    max_servers=None,
    up_thresh_ratio=0.8,
    down_thresh_ratio=0.3,
):
    min_servers = min_servers or max(2, NUM_SERVERS // 2)
    max_servers = max_servers or NUM_SERVERS
    ring = BoundedHashRing_RJ_CH(
        capacity=SERVER_CAPACITY, vnodes=NUM_VIRTUAL_NODES, max_attempts=50
    )
    # start with a small cluster
    for i in range(min_servers):
        ring.add_server(f"S{i}")
    next_server_id = min_servers
    autoscale = {"added": 0, "removed": 0}
    total_attempts = 0

    for k in keys:
        _, cost = ring.assign_key(k)
        total_attempts += cost

        # autoscale up
        loads = list(ring.server_loads.values()) or [0]
        if (
            max(loads) >= math.ceil(SERVER_CAPACITY * up_thresh_ratio)
            and len(ring.server_loads) < max_servers
        ):
            ring.add_server(f"S{next_server_id}")
            next_server_id += 1
            autoscale["added"] += 1

        # autoscale down
        loads_items = list(ring.server_loads.items())
        if (
            loads_items
            and all(
                l < math.floor(SERVER_CAPACITY * down_thresh_ratio)
                for _, l in loads_items
            )
            and len(ring.server_loads) > min_servers
        ):
            to_remove = min(ring.server_loads.items(), key=lambda kv: kv[1])[0]
            ring.remove_server(to_remove)
            autoscale["removed"] += 1

    return _detailed_stats_from_ring(ring, len(keys), total_attempts, autoscale)


def run_all():
    keys = generate_uniform_keys(NUM_KEYS_UNIFORM)

    fixed_ch_stats = run_fixed_k_ch(keys)
    fixed_rj_stats = run_fixed_k_rj(keys)
    dynamic_rj_stats = run_dynamic_k_rj(keys)

    print("Fixed-k CH (first-clockwise):", fixed_ch_stats)
    print("Fixed-k Random-Jump:", fixed_rj_stats)
    print("Dynamic-k Random-Jump (autoscale):", dynamic_rj_stats)


if __name__ == "__main__":
    run_all()
