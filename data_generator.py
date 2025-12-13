import numpy as np
from config import NUM_KEYS_UNIFORM, DLB_ZIPF_ALPHA


def generate_uniform_keys(num_keys=NUM_KEYS_UNIFORM):
    return [f"key_{i}" for i in range(num_keys)]


# We use this function to generate keys with Zipfian distribution, which simulates real world web requests,
# where a few keys are extremely popular and most are not.
def generate_zipfian_keys(num_keys, total_possible_keys=100000, alpha=DLB_ZIPF_ALPHA):
    a = alpha
    idx = np.random.zipf(a, size=num_keys).astype(int)
    idx = np.clip(idx, 1, total_possible_keys)
    return [f"key_{i}" for i in idx]


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


def generate_multiple_zipfian_datasets(
    num_datasets, num_keys, total_possible_keys=100000, alphas=None
):
    """
    Generate multiple datasets with Zipfian distributions.

    Args:
        num_datasets (int): Number of datasets to generate.
        num_keys (int): Number of keys in each dataset.
        total_possible_keys (int): Total unique keys.
        alphas (list): List of alpha values for each dataset. If None, use default alpha.

    Returns:
        list: List of datasets, where each dataset is a list of keys.
    """
    if alphas is None:
        alphas = [DLB_ZIPF_ALPHA] * num_datasets  # Use default alpha for all datasets
    elif len(alphas) != num_datasets:
        raise ValueError("The length of alphas must match num_datasets.")

    datasets = []
    for alpha in alphas:
        dataset = generate_zipfian_keys(num_keys, total_possible_keys, alpha)
        datasets.append(dataset)
    return datasets
