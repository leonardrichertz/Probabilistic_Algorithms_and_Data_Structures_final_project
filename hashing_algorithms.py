# hashing_algorithms.py

import hashlib
import bisect
from collections import defaultdict
from config import HASH_SPACE_SIZE, NUM_VIRTUAL_NODES, SERVER_CAPACITY, MAX_RJ_ATTEMPTS


class ConsistentHashRing:
    """Base class for Consistent Hashing structure."""

    def __init__(self, num_replicas=NUM_VIRTUAL_NODES, capacity=SERVER_CAPACITY):
        self.ring = []
        self.server_map = {}
        self.num_replicas = num_replicas
        self.capacity = capacity
        self.server_loads = defaultdict(int)
        self.key_assignments = {}

    def _hash(self, key):
        """Standard 32-bit hash function (SHA1 truncated)."""
        return int(hashlib.sha1(key.encode()).hexdigest(), 16) % HASH_SPACE_SIZE

    def add_server(self, server_name):
        """Adds a server (and its virtual replicas) to the ring."""
        self.server_loads[server_name] = 0
        for i in range(self.num_replicas):
            key = f"{server_name}#{i}"
            server_hash = self._hash(key)
            bisect.insort(self.ring, server_hash)
            self.server_map[server_hash] = server_name

    def remove_server(self, server_name):
        """Removes a server and returns the count of keys that need rehashing."""
        keys_to_reassign = [
            k for k, s in self.key_assignments.items() if s == server_name
        ]

        if server_name in self.server_loads:
            del self.server_loads[server_name]

        # Remove virtual nodes (simplistic removal, sorting is implicit via bisect)
        hashes_to_remove = [
            h for h, name in self.server_map.items() if name == server_name
        ]
        for h in hashes_to_remove:
            if h in self.ring:
                self.ring.remove(h)
            if h in self.server_map:
                del self.server_map[h]

        # Clear removed keys from assignments
        for key in keys_to_reassign:
            del self.key_assignments[key]

        return len(keys_to_reassign)

    def get_server_start_index(self, key):
        """Finds the starting point (closest server clockwise) for a key."""
        key_hash = self._hash(key)
        return bisect.bisect_left(self.ring, key_hash)


class BoundedHashRing_CH_BL(ConsistentHashRing):
    """Implements CH-BL with the deterministic clockwise search (cascading)."""

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.total_cascades = 0

    def assign_key(self, key):
        """Attempts to assign a key using the deterministic clockwise search."""
        if not self.ring:
            return None, 0

        start_idx = self.get_server_start_index(key)
        num_virtual_nodes = len(self.ring)
        search_count = 0

        for i in range(num_virtual_nodes):
            idx = (start_idx + i) % num_virtual_nodes
            server_hash = self.ring[idx]
            server_name = self.server_map[server_hash]

            if self.server_loads[server_name] < self.capacity:
                self.server_loads[server_name] += 1
                self.key_assignments[key] = server_name
                self.total_cascades += search_count
                return server_name, search_count

            search_count += 1

        # All servers are full
        return None, search_count


class BoundedHashRing_RJ_CH(ConsistentHashRing):
    """Implements RJ-CH using re-hashing with the attempt number to 'jump'."""

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.total_jumps = 0

    def assign_key(self, key):
        """Attempts to assign a key using RJ-CH's re-hashing ("jump") approach."""
        if not self.ring:
            return None, 0

        key_base = key

        for attempt in range(MAX_RJ_ATTEMPTS):
            # The 'jump' hash: key + attempt number
            current_key_name = f"{key_base}#{attempt}" if attempt > 0 else key_base
            key_hash = self._hash(current_key_name)

            # Find the starting server for this jump
            idx = bisect.bisect_left(self.ring, key_hash)
            if idx == len(self.ring):
                idx = 0

            server_hash = self.ring[idx]
            server_name = self.server_map[server_hash]

            if self.server_loads[server_name] < self.capacity:
                self.server_loads[server_name] += 1
                self.key_assignments[key_base] = server_name
                self.total_jumps += attempt
                return server_name, attempt

        # Max attempts reached
        return None, MAX_RJ_ATTEMPTS
