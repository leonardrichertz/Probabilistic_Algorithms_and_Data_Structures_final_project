# data_generator.py

import random
import numpy as np
from config import DLB_ZIPF_ALPHA


def generate_uniform_keys(num_keys):
    """Generates a list of uniformly distributed unique keys."""
    return [f"object_{i}" for i in range(num_keys)]


def generate_zipfian_keys(num_keys, total_possible_keys=100000):
    """
    Generates keys following a Zipf (skewed) distribution.
    A small number of 'hot keys' will be requested much more frequently.
    """
    # Keys are generated from a set of total_possible_keys

    # Generate Zipfian random numbers (indices)
    a = DLB_ZIPF_ALPHA  # Alpha > 1 means a heavy skew

    # We use size=num_keys samples from a distribution of total_possible_keys items
    # The output is a list of integers (indices)
    zipf_indices = np.random.zipf(a, num_keys).astype(int)

    # Clip to avoid errors if an index exceeds total_possible_keys
    zipf_indices = np.clip(zipf_indices, 1, total_possible_keys)

    # Convert indices back to key strings
    return [f"key_{i}" for i in zipf_indices]


def generate_training_data(num_samples, num_servers, capacity):
    """
    Generates training data (X, Y) for the DLB model.
    X: [Hashed Key ID, Load1, Load2, ...]
    Y: One-hot vector pointing to the ideal (min-loaded) server.

    Since training must be fast, this uses the MIN-LOAD strategy as the 'Ground Truth'.
    """

    X = []
    Y = []

    server_loads = np.zeros(num_servers)

    # Use a large number of unique key candidates to mimic a real environment
    keys = generate_zipfian_keys(num_samples * 2)

    for i in range(num_samples):
        key = keys[i]

        # 1. Input Features (X):
        # We need a stable representation of the key for the NN.
        # A simple one-hot encoding or hash is used here.
        key_id_feature = hash(key) % 10000  # Simplified key feature

        input_vector = [key_id_feature] + list(server_loads)
        X.append(input_vector)

        # 2. Target Label (Y): The ideal assignment (least loaded server)
        min_load_index = np.argmin(server_loads)
        target_vector = np.zeros(num_servers)
        target_vector[min_load_index] = 1
        Y.append(target_vector)

        # 3. Simulate assignment and update load (critical for training realism)
        server_loads[min_load_index] += 1

        # Reset server loads occasionally to prevent saturation and keep the load dynamics realistic
        if i % (num_servers * capacity) == 0 and i > 0:
            server_loads = np.zeros(num_servers)

    return np.array(X), np.array(Y)
