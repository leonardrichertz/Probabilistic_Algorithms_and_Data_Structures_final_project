import math
from config import NUM_SERVERS, SERVER_CAPACITY, NUM_KEYS_UNIFORM
from data_generator import generate_uniform_keys
from hashing_algorithms import BoundedHashRing_CH_BL, BoundedHashRing_RJ_CH


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


def run_fixed_k_ch(keys):
    ring = BoundedHashRing_CH_BL(capacity=SERVER_CAPACITY)
    for i in range(NUM_SERVERS):
        ring.add_server(f"S{i}")
    for k in keys:
        ring.assign_key(k)
    return _stats_from_ring(ring, len(keys))


def run_fixed_k_rj(keys):
    ring = BoundedHashRing_RJ_CH(capacity=SERVER_CAPACITY, max_attempts=50)
    for i in range(NUM_SERVERS):
        ring.add_server(f"S{i}")
    for k in keys:
        ring.assign_key(k)
    return _stats_from_ring(ring, len(keys))


def run_dynamic_k_rj(
    keys,
    min_servers=None,
    max_servers=None,
    up_thresh_ratio=0.8,
    down_thresh_ratio=0.3,
):
    min_servers = min_servers or max(2, NUM_SERVERS // 2)
    max_servers = max_servers or NUM_SERVERS
    ring = BoundedHashRing_RJ_CH(capacity=SERVER_CAPACITY, max_attempts=50)
    # start with a small cluster
    for i in range(min_servers):
        ring.add_server(f"S{i}")
    next_server_id = min_servers

    for k in keys:
        ring.assign_key(k)
        # autoscale up: if any server >= up threshold and we can add
        loads = list(ring.server_loads.values()) or [0]
        if (
            max(loads) >= math.ceil(SERVER_CAPACITY * up_thresh_ratio)
            and len(ring.server_loads) < max_servers
        ):
            ring.add_server(f"S{next_server_id}")
            next_server_id += 1
        # autoscale down: if all servers below down threshold and we have > min
        loads = list(ring.server_loads.items())
        if (
            loads
            and all(
                l < math.floor(SERVER_CAPACITY * down_thresh_ratio) for _, l in loads
            )
            and len(ring.server_loads) > min_servers
        ):
            # remove the least-loaded server
            to_remove = min(ring.server_loads.items(), key=lambda kv: kv[1])[0]
            ring.remove_server(to_remove)

    return _stats_from_ring(ring, len(keys))


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
