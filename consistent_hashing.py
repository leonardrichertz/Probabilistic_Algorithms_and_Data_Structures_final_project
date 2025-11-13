# ...existing code...
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
        return int(hashlib.sha1(str(key).encode()).hexdigest(), 16) % HASH_SPACE_SIZE

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

        # Remove virtual nodes
        hashes_to_remove = [
            h for h, name in list(self.server_map.items()) if name == server_name
        ]
        for h in hashes_to_remove:
            if h in self.ring:
                self.ring.remove(h)
            if h in self.server_map:
                del self.server_map[h]

        # Clear removed keys from assignments
        for key in keys_to_reassign:
            if key in self.key_assignments:
                del self.key_assignments[key]

        return len(keys_to_reassign)

    def get_server_start_index(self, key):
        """Finds the starting point (closest server clockwise) for a key."""
        key_hash = self._hash(key)
        return bisect.bisect_left(self.ring, key_hash)

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
        h = self._hash(key)
        if not self.ring:
            return None, 0
        # start at chosen vnode and walk clockwise until free server found
        start = bisect.bisect_left(self.ring, h)
        n = len(self.ring)
        steps = 0
        for i in range(n):
            idx = (start + i) % n
            server = self.server_map[self.ring[idx]]
            if self.server_loads[server] < self.capacity:
                self.server_loads[server] += 1
                self.key_assignments[key] = server
                return server, steps
            steps += 1
        return None, steps


class BoundedHashRing_RJ_CH(ConsistentHashRing):
    """Random-Jump: try hashed jumps 'key#attempt' up to max_attempts."""

    def __init__(
        self,
        capacity=SERVER_CAPACITY,
        vnodes=NUM_VIRTUAL_NODES,
        max_attempts=MAX_RJ_ATTEMPTS,
    ):
        super().__init__(num_replicas=vnodes, capacity=capacity)
        self.max_attempts = max_attempts

    def assign_key(self, key):
        if not self.ring:
            return None, 0
        # first try normal placement
        server = self.get_server_for_hash(self._hash(key))
        if server and self.server_loads[server] < self.capacity:
            self.server_loads[server] += 1
            self.key_assignments[key] = server
            return server, 0
        # random-jump attempts
        for attempt in range(1, self.max_attempts + 1):
            attempt_key = f"{key}#{attempt}"
            server = self.get_server_for_hash(self._hash(attempt_key))
            if server and self.server_loads[server] < self.capacity:
                self.server_loads[server] += 1
                self.key_assignments[key] = server
                return server, attempt
        return None, self.max_attempts


class BoundedHashRing_RehashThreshold(ConsistentHashRing):
    """Rehash when chosen server load >= threshold_ratio * capacity."""

    def __init__(
        self,
        capacity=SERVER_CAPACITY,
        vnodes=NUM_VIRTUAL_NODES,
        threshold_ratio=0.8,
        max_attempts=MAX_RJ_ATTEMPTS,
    ):
        super().__init__(num_replicas=vnodes, capacity=capacity)
        self.threshold_ratio = threshold_ratio
        self.max_attempts = max_attempts

    def assign_key(self, key):
        if not self.ring:
            return None, 0
        capacity_threshold = int(self.capacity * self.threshold_ratio)
        server = self.get_server_for_hash(self._hash(key))
        if server and self.server_loads[server] < capacity_threshold:
            self.server_loads[server] += 1
            self.key_assignments[key] = server
            return server, 0
        # try rehash attempts
        for attempt in range(1, self.max_attempts + 1):
            s = self.get_server_for_hash(self._hash(f"{key}#{attempt}"))
            if s and self.server_loads[s] < capacity_threshold:
                self.server_loads[s] += 1
                self.key_assignments[key] = s
                return s, attempt
        # fallback: least-loaded server with free capacity
        candidates = [srv for srv, l in self.server_loads.items() if l < self.capacity]
        if candidates:
            least = min(candidates, key=lambda s: self.server_loads[s])
            self.server_loads[least] += 1
            self.key_assignments[key] = least
            return least, self.max_attempts
        return None, self.max_attempts
