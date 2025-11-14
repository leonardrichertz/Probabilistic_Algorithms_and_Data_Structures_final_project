import math
from config import NUM_SERVERS, SERVER_CAPACITY, NUM_KEYS_UNIFORM, NUM_VIRTUAL_NODES
from data_generator import generate_uniform_keys
from hashing_algorithms import BoundedHashRing_CH_BL, BoundedHashRing_RJ_CH
import statistics


# The jain index is a common metric for measuring load balance. It measures how evenly the load is distributed across servers.
# A jain index of 1 indicates perfect balance (all servers have the same load), while lower values indicate more imbalance.
def _jain_index(loads):
    if not loads:
        return 0.0
    s = sum(loads)
    s2 = sum(l * l for l in loads)
    n = len(loads)
    return (s * s) / (n * s2) if s2 > 0 else 0.0


def _snapshot(ring):
    loads = list(ring.server_loads.values()) or [0]
    n = len(loads)
    avg = sum(loads) / n if loads else 0
    mx = max(loads) if loads else 0
    percent_full = sum(1 for l in loads if l >= ring.capacity) / max(1, n)
    return {
        "num_servers": n,
        "avg_load": avg,
        "max_load": mx,
        "percent_full": percent_full,
        "jain": _jain_index(loads),
    }


def run_spike_experiment(
    steady_keys=500,
    spike_keys=2000,
    snapshot_interval=250,
    up_thresh=0.8,
    down_thresh=0.3,
):
    # generate simple keystreams: steady then spike (same per-key rate modeled by counts)
    steady = generate_uniform_keys(steady_keys)
    spike = generate_uniform_keys(
        spike_keys
    )  # use same distribution; you can use Zipf for hotspots
    keys = steady + spike

    # fixed CH
    fixed = BoundedHashRing_CH_BL(capacity=SERVER_CAPACITY)
    for i in range(NUM_SERVERS):
        fixed.add_server(f"S{i}")
    fixed_tseries = []
    for i, k in enumerate(keys, 1):
        fixed.assign_key(k)
        if i % snapshot_interval == 0 or i == len(steady) or i == len(keys):
            fixed_tseries.append((i, _snapshot(fixed)))

    # fixed RJ
    fixed_rj = BoundedHashRing_RJ_CH(
        capacity=SERVER_CAPACITY, vnodes=NUM_VIRTUAL_NODES, max_attempts=50
    )
    for i in range(NUM_SERVERS):
        fixed_rj.add_server(f"S{i}")
    fixed_rj_tseries = []
    for i, k in enumerate(keys, 1):
        fixed_rj.assign_key(k)
        if i % snapshot_interval == 0 or i == len(steady) or i == len(keys):
            fixed_rj_tseries.append((i, _snapshot(fixed_rj)))

    # dynamic RJ (autoscale up when any server >= up_thresh*capacity)
    min_servers = max(2, NUM_SERVERS // 2)
    max_servers = NUM_SERVERS * 2
    dyn = BoundedHashRing_RJ_CH(
        capacity=SERVER_CAPACITY, vnodes=NUM_VIRTUAL_NODES, max_attempts=50
    )
    for i in range(min_servers):
        dyn.add_server(f"S{i}")
    next_id = min_servers
    autoscale = {"added": 0, "removed": 0}
    dyn_tseries = []
    for i, k in enumerate(keys, 1):
        dyn.assign_key(k)
        loads = list(dyn.server_loads.values()) or [0]
        # simple immediate up-scale policy
        if (
            max(loads) >= math.ceil(SERVER_CAPACITY * up_thresh)
            and len(dyn.server_loads) < max_servers
        ):
            dyn.add_server(f"S{next_id}")
            next_id += 1
            autoscale["added"] += 1
        # simple down-scale policy (only when all very low)
        if (
            all(l < math.floor(SERVER_CAPACITY * down_thresh) for l in loads)
            and len(dyn.server_loads) > min_servers
        ):
            to_remove = min(dyn.server_loads.items(), key=lambda kv: kv[1])[0]
            dyn.remove_server(to_remove)
            autoscale["removed"] += 1
        if i % snapshot_interval == 0 or i == len(steady) or i == len(keys):
            snap = _snapshot(dyn)
            snap["autoscale"] = dict(autoscale)
            dyn_tseries.append((i, snap))

    # print concise time series
    def print_series(name, series):
        print(f"\n{name} snapshots (index, num_servers, avg, max, %full, jain):")
        for idx, s in series:
            print(
                f"{idx:6d} | {s['num_servers']:2d} | avg {s['avg_load']:.2f} | max {s['max_load']:2d} | %full {s['percent_full']:.2f} | jain {s['jain']:.3f}"
                + (
                    f" | autoscale +{s.get('autoscale',{}).get('added',0)}"
                    if "autoscale" in s
                    else ""
                )
            )

    print_series("Fixed-CH", fixed_tseries)
    print_series("Fixed-RJ", fixed_rj_tseries)
    print_series("Dynamic-RJ", dyn_tseries)


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
    run_spike_experiment(steady_keys=500, spike_keys=3000, snapshot_interval=500)

    from createplots import plot_stats_comparison

    plot_stats_comparison(
        [fixed_ch_stats, fixed_rj_stats, dynamic_rj_stats],
        labels=["Fixed-k CH", "Fixed-k RJ", "Dynamic-k RJ"],
        out_prefix="hashing_comparison",
    )

    print("Fixed-k CH (first-clockwise):", fixed_ch_stats)
    print("Fixed-k Random-Jump:", fixed_rj_stats)
    print("Dynamic-k Random-Jump (autoscale):", dynamic_rj_stats)


if __name__ == "__main__":
    run_all()
