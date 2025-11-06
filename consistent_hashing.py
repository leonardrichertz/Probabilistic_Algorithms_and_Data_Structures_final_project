import hashlib
import bisect
import random
from collections import defaultdict

# --- CONFIGURATION CONSTANTS ---
HASH_SPACE_SIZE = 2**32  # Total size of the hash ring
NUM_VIRTUAL_NODES = 100  # Virtual nodes per physical server for better balance
SERVER_CAPACITY = 1000  # Bounded Load capacity for the simulation

# --- BASE CONSISTENT HASHING CLASS ---


class ConsistentHashRing:
    """
    Base class for Consistent Hashing structure.
    Manages the hash ring, server placement, and basic hashing logic.
    """

    def __init__(self, num_replicas=NUM_VIRTUAL_NODES, capacity=SERVER_CAPACITY):
        self.ring = []
        self.server_map = {}  # Maps hash point to physical server name
        self.num_replicas = num_replicas
        self.capacity = capacity
        self.server_loads = defaultdict(
            int
        )  # Tracks current load for each physical server
        self.key_assignments = {}  # Tracks where each key is stored

    def _hash(self, key):
        """Standard 32-bit hash function (SHA1 truncated)."""
        return int(hashlib.sha1(key.encode()).hexdigest(), 16) % HASH_SPACE_SIZE

    def add_server(self, server_name):
        """Adds a server (and its virtual replicas) to the ring."""
        # Reset server load if re-adding
        self.server_loads[server_name] = 0

        for i in range(self.num_replicas):
            key = f"{server_name}#{i}"
            server_hash = self._hash(key)

            # Insert and maintain sorted order
            bisect.insort(self.ring, server_hash)
            self.server_map[server_hash] = server_name

    def remove_server(self, server_name):
        """Removes a server and all its replicas."""
        if server_name in self.server_loads:
            del self.server_loads[server_name]

        # Efficient removal of virtual nodes and key assignments would be implemented here.
        # For simplicity, we just remove virtual nodes and clear assignments for now.
        hashes_to_remove = [
            h for h, name in self.server_map.items() if name == server_name
        ]

        for h in hashes_to_remove:
            del self.server_map[h]
            self.ring.remove(h)

        # Reassign keys previously held by this server (key rehashing metric)
        keys_to_reassign = [
            k for k, s in self.key_assignments.items() if s == server_name
        ]
        for key in keys_to_reassign:
            del self.key_assignments[key]
            # In a real scenario, these keys would be re-assigned.

        return len(keys_to_reassign)  # Return the count of rehashed keys

    def get_server_start_index(self, key):
        """Finds the starting point (closest server clockwise) for a key."""
        key_hash = self._hash(key)
        # bisect_left returns the index of the first item >= key_hash
        return bisect.bisect_left(self.ring, key_hash)


# --- ALGORITHM 1: CONSISTENT HASHING WITH BOUNDED LOADS (CH-BL) ---


class BoundedHashRing_CH_BL(ConsistentHashRing):
    """
    Implements CH-BL with the deterministic clockwise search upon overflow.
    This implementation will demonstrate the CASCADED OVERFLOW problem.
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.total_cascades = 0  # Metric to track the severity of cascading

    def assign_key(self, key):
        """
        Attempts to assign a key using the deterministic clockwise search.
        """
        key_hash = self._hash(key)
        start_idx = self.get_server_start_index(key)
        num_virtual_nodes = len(self.ring)

        # Start search counter (0 means first choice was available)
        search_count = 0

        # Loop up to the number of virtual nodes (full circle check)
        for i in range(num_virtual_nodes):
            idx = (start_idx + i) % num_virtual_nodes
            server_hash = self.ring[idx]
            server_name = self.server_map[server_hash]

            if self.server_loads[server_name] < self.capacity:

                # ASSIGNMENT SUCCESS
                self.server_loads[server_name] += 1
                self.key_assignments[key] = server_name
                self.total_cascades += search_count
                return server_name, search_count

            # If server is full, increment search_count for the metric
            search_count += 1

        # All servers are full
        return None, search_count


# --- ALGORITHM 2: RANDOM JUMP CONSISTENT HASHING (RJ-CH) ---


class BoundedHashRing_RJ_CH(ConsistentHashRing):
    """
    Implements RJ-CH using re-hashing with the attempt number to 'jump'
    to a new, random location upon overflow.
    This implementation will mitigate the CASCADED OVERFLOW problem.
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.max_attempts = 100  # Maximum jumps before giving up (practical limit)
        self.total_jumps = 0  # Metric to track the number of jumps needed

    def assign_key(self, key):
        """
        Attempts to assign a key using RJ-CH's re-hashing ("jump") approach.
        """
        key_base = key

        # Loop up to max_attempts to find an available server
        for attempt in range(self.max_attempts):

            # 1. Determine the hash point for the current attempt (The "Jump")
            # The first attempt uses the key itself. Subsequent attempts use the key + attempt number.
            if attempt > 0:
                key = f"{key_base}#{attempt}"

            key_hash = self._hash(key)

            # 2. Find the starting server for this jump (Standard CH lookup)
            idx = bisect.bisect_left(self.ring, key_hash)
            if idx == len(self.ring):
                idx = 0

            server_hash = self.ring[idx]
            server_name = self.server_map[server_hash]

            # 3. Check Capacity
            if self.server_loads[server_name] < self.capacity:

                # ASSIGNMENT SUCCESS
                self.server_loads[server_name] += 1
                self.key_assignments[key_base] = server_name
                self.total_jumps += attempt
                return server_name, attempt

        # Max attempts reached, system is likely overloaded
        return None, self.max_attempts


# --- SIMULATION AND COMPARISON ---


def simulate_load(hashing_ring, num_keys):
    """Runs a load simulation and records metrics."""
    # Reset metrics
    hashing_ring.server_loads = defaultdict(int)
    hashing_ring.key_assignments = {}

    if hasattr(hashing_ring, "total_cascades"):
        hashing_ring.total_cascades = 0
    if hasattr(hashing_ring, "total_jumps"):
        hashing_ring.total_jumps = 0

    total_assigned = 0
    total_search_cost = 0

    for i in range(num_keys):
        key = f"object_{i}"

        server, cost = hashing_ring.assign_key(key)

        if server:
            total_assigned += 1
            total_search_cost += cost
        else:
            # System overload; stop
            break

    # Calculate load variance
    loads = list(hashing_ring.server_loads.values())
    avg_load = sum(loads) / len(loads)
    variance = sum((l - avg_load) ** 2 for l in loads) / len(loads)

    return {
        "assigned_keys": total_assigned,
        "avg_search_cost": total_search_cost / max(1, total_assigned),
        "load_variance": variance,
        "full_servers": sum(1 for load in loads if load == hashing_ring.capacity),
        "server_loads": loads,
    }


# --- EXECUTE SIMULATION ---

if __name__ == "__main__":

    print("--- Consistent Hashing Comparison Simulation ---")

    # Initialize the rings
    servers = [f"Server-{i}" for i in range(10)]  # 10 physical servers

    # CH-BL (The problem case)
    ch_bl_ring = BoundedHashRing_CH_BL(capacity=SERVER_CAPACITY)
    for s in servers:
        ch_bl_ring.add_server(s)

    # RJ-CH (The proposed solution)
    rj_ch_ring = BoundedHashRing_RJ_CH(capacity=SERVER_CAPACITY)
    for s in servers:
        rj_ch_ring.add_server(s)

    NUM_KEYS = 250000  # Total keys to assign (10 servers * 50 capacity = 500 max)

    print(
        f"\nConfiguration: {len(servers)} Servers, Capacity: {SERVER_CAPACITY} keys each."
    )
    print(f"Assigning {NUM_KEYS} total keys (Full Capacity)...\n")

    # Run CH-BL
    results_ch_bl = simulate_load(ch_bl_ring, NUM_KEYS)

    # Run RJ-CH
    results_rj_ch = simulate_load(rj_ch_ring, NUM_KEYS)

    print("-" * 40)

    # --- ANALYSIS ---

    print("ALGORITHM 1: Consistent Hashing with Bounded Loads (CH-BL)")
    print(f"Total Keys Assigned: {results_ch_bl['assigned_keys']}")
    print(
        f"Servers that reached capacity (Overloaded): {results_ch_bl['full_servers']}"
    )
    print(
        f"Avg. Search Cost (Cascading Severity): {results_ch_bl['avg_search_cost']:.2f} virtual nodes checked per key."
    )
    print(f"Load Variance (Unevenness): {results_ch_bl['load_variance']:.2f}\n")

    print("ALGORITHM 2: Random Jump Consistent Hashing (RJ-CH)")
    print(f"Total Keys Assigned: {results_rj_ch['assigned_keys']}")
    print(
        f"Servers that reached capacity (Overloaded): {results_rj_ch['full_servers']}"
    )
    print(
        f"Avg. Search Cost (Jump Attempts): {results_rj_ch['avg_search_cost']:.2f} attempts needed per key."
    )
    print(f"Load Variance (Evenness): {results_rj_ch['load_variance']:.2f}")
    print("-" * 40)

    print("\n--- CONCLUSION ---")

    # Explanation of results
    if results_ch_bl["load_variance"] > results_rj_ch["load_variance"] * 1.5:
        print("RJ-CH successfully mitigated cascading failure!")
        print(
            f"CH-BL's load variance ({results_ch_bl['load_variance']:.2f}) is significantly higher."
        )
        print(
            "The high search cost (cascades) in CH-BL means objects skipped full servers and disproportionately overloaded the next available one."
        )
        print(
            "RJ-CH's low search cost (jump attempts) and low load variance show that overflow load was distributed uniformly, as predicted by the paper."
        )
    else:
        print(
            "Results are close, suggesting the number of keys or servers needs to be increased to fully demonstrate the effect."
        )
        print("Try increasing SERVER_CAPACITY or NUM_KEYS.")
