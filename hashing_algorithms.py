import hashlib
import bisect
from collections import defaultdict
from config import HASH_SPACE_SIZE, SERVER_CAPACITY, K


class ConsistentHashRing:
    """Base class for Consistent Hashing structure."""

    def __init__(self, capacity=SERVER_CAPACITY):
        self.ring = []
        self.server_map = {}
        self.capacity = capacity
        self.server_loads = defaultdict(int)
        self.key_assignments = {}

    def _hash(self, key):
        """Standard 32-bit hash function (SHA1 truncated)."""
        return int(hashlib.sha1(str(key).encode()).hexdigest(), 16) % HASH_SPACE_SIZE

    def add_server(self, server_name):
        """Adds a server to the ring."""
        self.server_loads[server_name] = 0
        server_hash = self._hash(server_name)
        bisect.insort(self.ring, server_hash)
        self.server_map[server_hash] = server_name

    def remove_server(self, server_name):
        """Removes a server and returns the count of keys that need rehashing."""
        keys_to_reassign = [
            k for k, s in self.key_assignments.items() if s == server_name
        ]

        if server_name in self.server_loads:
            del self.server_loads[server_name]

        hashes_to_remove = [
            h for h, name in list(self.server_map.items()) if name == server_name
        ]
        for h in hashes_to_remove:
            if h in self.ring:
                self.ring.remove(h)
            if h in self.server_map:
                del self.server_map[h]

        for key in keys_to_reassign:
            if key in self.key_assignments:
                del self.key_assignments[key]

        return len(keys_to_reassign)

    def get_server_for_hash(self, h):
        """Return server name for a given hash value (wraps around)."""
        if not self.ring:
            return None
        idx = bisect.bisect_left(self.ring, h)
        if idx == len(self.ring):
            idx = 0
        return self.server_map[self.ring[idx]]


class BoundedHashRing_CH_BL(ConsistentHashRing):
    """Fixed-k CH: assign to first clockwise server with free capacity."""

    def assign_key(self, key):
        total_hashes = 0
        h = self._hash(key)
        if not self.ring:
            return None, 0, total_hashes
        start = bisect.bisect_left(self.ring, h)
        n = len(self.ring)
        steps = 0
        for i in range(n):
            idx = (start + i) % n
            server = self.server_map[self.ring[idx]]
            total_hashes += 1  # Increment the counter
            if self.server_loads[server] < self.capacity:
                self.server_loads[server] += 1
                self.key_assignments[key] = server
                return server, steps, total_hashes
            steps += 1
        return None, steps, total_hashes


class BoundedHashRing_RJ_CH(ConsistentHashRing):
    """Random-Jump: try hashed jumps 'key#attempt' up to k."""

    def __init__(
        self,
        capacity=SERVER_CAPACITY,
        k=K,
    ):
        super().__init__(capacity=capacity)
        self.k = k

    def assign_key(self, key):
        total_hashes = 0
        if not self.ring:
            return None, 0, total_hashes

        # Step 1: Generate k random servers
        for attempt in range(1, self.k + 1):
            total_hashes += 1  # Increment the global counter
            server = self.get_server_for_hash(self._hash(f"{key}#{attempt}"))
            if server and self.server_loads[server] < self.capacity:
                self.server_loads[server] += 1
                self.key_assignments[key] = server
                return server, attempt, total_hashes

        # Step 2: If no server meets the capacity, return None
        return None, self.k, total_hashes


class BoundedHashRing_RehashThreshold(ConsistentHashRing):
    """Rehash when chosen server load >= threshold_ratio * capacity."""

    def __init__(
        self,
        capacity=SERVER_CAPACITY,
        threshold_ratio=0.8,
        k=K,
    ):
        super().__init__(capacity=capacity)
        self.threshold_ratio = threshold_ratio
        self.k = k
        self.last_request_time = {}
        self.current_time = 0

    def assign_key(self, key):
        total_hashes = 0
        if not self.ring:
            return None, 0, total_hashes

        capacity_threshold = int(self.capacity * self.threshold_ratio)
        k_candidates = []

        # Step 1: Generate k random servers
        for attempt in range(1, self.k + 1):
            total_hashes += 1  # Increment the global counter
            candidate_server = self.get_server_for_hash(self._hash(f"{key}#{attempt}"))
            if candidate_server:
                k_candidates.append(candidate_server)

        # Step 2: Find the least loaded server among the k candidates
        best_server = None
        min_load = float("inf")
        for server in k_candidates:

            if self.server_loads[server] < min_load:
                best_server = server
                min_load = self.server_loads[server]

        # Step 3: Check if the best server meets the threshold
        if best_server and self.server_loads[best_server] < capacity_threshold:
            self.server_loads[best_server] += 1
            self.key_assignments[key] = best_server
            return best_server, len(k_candidates), total_hashes

        # Step 4: If no server meets the threshold, keep hashing until we find one or hit max attempts
        attempt = self.k + 1
        max_attempts = 100  # Set a maximum number of attempts
        while attempt <= max_attempts:
            new_server = self.get_server_for_hash(self._hash(f"{key}#{attempt}"))
            if new_server and self.server_loads[new_server] < capacity_threshold:
                self.server_loads[new_server] += 1
                self.key_assignments[key] = new_server
                total_hashes += 1  # Increment the global counter
                return new_server, attempt, total_hashes
            attempt += 1

        return None, attempt, total_hashes
