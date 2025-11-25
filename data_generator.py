import numpy as np
from config import NUM_KEYS_UNIFORM, DLB_ZIPF_ALPHA


def generate_uniform_keys(num_keys=NUM_KEYS_UNIFORM):
    return [f"key_{i}" for i in range(num_keys)]


# We use this function to generate keys with Zipfian distribution, which simulates real world web requests,
# where a few keys are extremely popular and most are not.
def generate_zipfian_keys(num_keys, total_possible_keys=100000):
    a = DLB_ZIPF_ALPHA
    idx = np.random.zipf(a, size=num_keys).astype(int)
    idx = np.clip(idx, 1, total_possible_keys)
    return [f"key_{i}" for i in idx]


# optional: tiny generator used by DLB training (if needed)
def generate_training_data(size, num_servers, server_capacity):
    if num_servers <= 0:
        raise ValueError("num_servers must be >= 1")
    server_loads = np.zeros(num_servers, dtype=int)
    X, Y = [], []
    for _ in range(size):
        # trivial feature: current loads -> one-hot for least-loaded
        features = server_loads.copy()
        label = np.zeros(num_servers, dtype=int)
        label[int(np.argmin(server_loads))] = 1
        X.append(np.concatenate(([0], features)))  # 0 placeholder for key-feature
        Y.append(label)
        # simulate an assignment
        server_loads[int(np.argmin(server_loads))] += 1
        server_loads = np.minimum(server_loads, server_capacity)
    return np.array(X, dtype=float), np.array(Y, dtype=int)
