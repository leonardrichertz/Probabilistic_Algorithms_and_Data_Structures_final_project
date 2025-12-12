import hashlib
import bisect
from collections import defaultdict
from config import HASH_SPACE_SIZE, NUM_VIRTUAL_NODES, SERVER_CAPACITY, MAX_RJ_ATTEMPTS


# TODO: Do we assign the query only to the servers that actually serve the website such as amazon.com
# Right now we simply assign to a free server and assume that it can serve the correct website??
# Is this needed to be changed?
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
        threshold_ratio=0.8,
        max_attempts=MAX_RJ_ATTEMPTS,
    ):
        super().__init__(num_replicas=vnodes, capacity=capacity)
        self.max_attempts = max_attempts
        self.threshold_ratio = threshold_ratio

    def assign_key(self, key):
        if not self.ring:
            return None, 0
        capacity_threshold = int(self.capacity * self.threshold_ratio)
        server = self.get_server_for_hash(self._hash(key))
        if server and self.server_loads[server] < capacity_threshold:
            self.server_loads[server] += 1
            self.key_assignments[key] = server
            return server, 0
        for attempt in range(1, self.max_attempts + 1):
            s = self.get_server_for_hash(self._hash(f"{key}#{attempt}"))
            if s and self.server_loads[s] < capacity_threshold:
                self.server_loads[s] += 1
                self.key_assignments[key] = s
                return s, attempt
        candidates = [srv for srv, l in self.server_loads.items() if l < self.capacity]
        if candidates:
            least = min(candidates, key=lambda s: self.server_loads[s])
            self.server_loads[least] += 1
            self.key_assignments[key] = least
            return least, self.max_attempts
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
        self.last_request_time = {}
        self.current_time = 0

    def assign_key(self, key):
        if not self.ring:
            return None, 0
        capacity_threshold = int(self.capacity * self.threshold_ratio)
        server = self.get_server_for_hash(self._hash(key))
        if server and self.server_loads[server] < capacity_threshold:
            self.server_loads[server] += 1
            self.key_assignments[key] = server
            return server, 0
        for attempt in range(1, self.max_attempts + 1):
            s = self.get_server_for_hash(self._hash(f"{key}#{attempt}"))
            if s and self.server_loads[s] < capacity_threshold:
                self.server_loads[s] += 1
                self.key_assignments[key] = s
                return s, attempt
        candidates = [
            srv for srv, l in self.server_loads.items() if l < capacity_threshold
        ]
        if candidates:
            least = min(candidates, key=lambda s: self.server_loads[s])
            self.server_loads[least] += 1
            self.key_assignments[key] = least
            return least, self.max_attempts
        # Autoscale: add a new server
        new_server_name = f"S{len(self.server_loads)}"
        print(f"Autoscaling: Adding new server {new_server_name}")
        self.add_server(new_server_name)

        self.rebalance_keys(new_server_name)

        return self.assign_key(key)

    def rebalance_keys(self, new_server):
        """Redistribute keys to balance load after adding a new server."""
        for key, server in list(self.key_assignments.items()):
            # Check if the current server is overloaded
            if self.server_loads[server] > self.capacity * self.threshold_ratio:
                # Remove the key from the current server
                self.delete_key(key)
                # Reassign the key to the new server
                self.assign_key(key)

    def get_natural_server_for_key(self, key):
        """Find the server this key would naturally hash to (first clockwise)."""
        if not self.ring:
            return None
        h = self._hash(key)
        return self.get_server_for_hash(h)

    def shutdown_idle_servers(self, idle_threshold):
        """
        Shut down servers that have been idle for more than `idle_threshold` time steps.
        Args:
            idle_threshold (int): Number of time steps a server can be idle before being shut down.
        Returns:
            list: Names of servers that were shut down.
        """
        idle_servers = [
            server
            for server, last_time in self.last_request_time.items()
            if self.current_time - last_time > idle_threshold
        ]
        for server in idle_servers:
            self.remove_server(server)
        return idle_servers

    # TODO: Delete item function, if the initial k servers have enough capacity, we can delete the key from the server that we created if we
    # had to rehash it to another server. This would free up space on the original server. So we delete the key and then create a new one as soon as the # key is deleted.
    # Go over these function again.
    def delete_key(self, key):
        """Remove a key assignment and free up server capacity."""
        if key in self.key_assignments:
            server = self.key_assignments[key]
            if server in self.server_loads:
                self.server_loads[server] -= 1
            del self.key_assignments[key]
            return server
        return None

    def get_keys_on_server(self, server_name):
        """Get all keys currently assigned to a specific server."""
        return [
            key for key, server in self.key_assignments.items() if server == server_name
        ]

    # Can we turn off a server afer a certain time of it being underutilized?
    def remove_server(self, server_name):
        """Removes a server and rehashes its keys."""
        keys_to_rehash = self.get_keys_on_server(server_name)

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

        for key in keys_to_rehash:
            # Compute the hash of the key
            key_hash = self._hash(key)
            # Find the next server clockwise
            new_server = self.get_server_for_hash(key_hash)
            if new_server and self.server_loads[new_server] < self.capacity:
                # Assign the key to the new server
                self.key_assignments[key] = new_server
                self.server_loads[new_server] += 1
            else:
                # Key cannot be reassigned due to capacity constraints
                del self.key_assignments[key]

        return len(keys_to_rehash)
