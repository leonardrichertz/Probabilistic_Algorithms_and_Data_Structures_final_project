# dlb_model.py

import numpy as np
import tensorflow as tf
from tensorflow import keras
from hashing_algorithms import ConsistentHashRing
from config import NUM_SERVERS, DLB_EPOCHS, DLB_TRAINING_SIZE, SERVER_CAPACITY
from data_generator import generate_training_data

# Disable TensorFlow warnings for a cleaner simulation output
tf.get_logger().setLevel("ERROR")


def create_dlb_model(num_servers):
    """Defines and compiles the simple Deep Learning Based Load Balancing model."""

    # Input: 1 (Key Feature) + NUM_SERVERS (Load Vector)
    input_size = num_servers + 1

    model = keras.Sequential(
        [
            keras.Input(shape=(input_size,)),
            keras.layers.Dense(64, activation="relu"),
            keras.layers.Dense(32, activation="relu"),
            # Output is a probability for each server (NUM_SERVERS classes)
            keras.layers.Dense(num_servers, activation="softmax"),
        ]
    )

    # Loss: Categorical Cross-Entropy (standard for multi-class classification)
    model.compile(
        optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"]
    )
    return model


def train_dlb_model(model, num_servers):
    """Generates training data and trains the model."""
    print(f"\n[DLB] Generating {DLB_TRAINING_SIZE} training samples for the DNN...")
    X_train, Y_train = generate_training_data(
        DLB_TRAINING_SIZE, num_servers, SERVER_CAPACITY
    )

    print(f"[DLB] Training Model for {DLB_EPOCHS} epochs...")
    # Training the model to learn the policy of 'picking the least loaded server'
    model.fit(X_train, Y_train, epochs=DLB_EPOCHS, verbose=0)
    print("[DLB] Training complete.")


class LearnedHashRing_DLB(ConsistentHashRing):
    """
    Implements the Learned Hashing Policy by using a trained DNN
    to make load-aware assignment decisions.
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.num_servers = len(self.server_loads)
        self.model = create_dlb_model(self.num_servers)
        self.server_names = list(self.server_loads.keys())
        train_dlb_model(self.model, self.num_servers)
        self.total_predictions = 0

    def assign_key(self, key):
        """
        Assigns a key by feeding the system state (loads) into the DNN
        and choosing the server with the highest predicted score.
        """
        if not self.server_loads:
            return None, 0

        # 1. Prepare Input Vector (X)
        key_id_feature = (
            hash(key) % 10000
        )  # Consistent key feature (as used in training)

        # Get current load vector
        load_vector = np.array([self.server_loads[s] for s in self.server_names])

        # Combine into input tensor: [key_feature, load1, load2, ...]
        input_vector = np.insert(load_vector, 0, key_id_feature)
        input_tensor = np.expand_dims(input_vector, axis=0)  # Add batch dimension

        # 2. DNN Inference (The Assignment Decision)
        prediction = self.model.predict(input_tensor, verbose=0)

        # The chosen server is the one with the highest probability/score
        chosen_server_index = np.argmax(prediction)
        chosen_server_name = self.server_names[chosen_server_index]
        self.total_predictions += 1

        # 3. Check Capacity (The DLB model is trained to minimize this failure,
        # but a check is still necessary in a bounded system)
        if self.server_loads[chosen_server_name] < self.capacity:
            self.server_loads[chosen_server_name] += 1
            self.key_assignments[key] = chosen_server_name
            # Search cost is 0 because the NN makes an immediate, load-aware decision
            return chosen_server_name, 0

        # Assignment failed (even the DLB model could not prevent overload)
        return None, 1  # Cost is 1 failure attempt
