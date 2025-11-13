import numpy as np
from tensorflow import keras
from config import DLB_EPOCHS, DLB_TRAINING_SIZE, SERVER_CAPACITY
from data_generator import generate_training_data


def create_dlb_model(num_servers):
    input_size = num_servers + 1
    model = keras.Sequential(
        [
            keras.Input(shape=(input_size,)),
            keras.layers.Dense(64, activation="relu"),
            keras.layers.Dense(32, activation="relu"),
            keras.layers.Dense(num_servers, activation="softmax"),
        ]
    )
    model.compile(
        optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"]
    )
    return model


def train_dlb_model(model, num_servers):
    X_train, Y_train = generate_training_data(
        DLB_TRAINING_SIZE, num_servers, SERVER_CAPACITY
    )
    model.fit(X_train, Y_train, epochs=DLB_EPOCHS, verbose=0)


class LearnedHashRing_DLB:
    """
    Minimal Learned (DLB) ring:
    - server_names: list of server identifiers (required)
    - capacity: per-server capacity
    - model: trained DNN used to pick a server given (key_feature + load_vector)
    """

    def __init__(self, server_names, capacity=SERVER_CAPACITY):
        if not server_names:
            raise ValueError("server_names must be a non-empty list")
        self.server_names = list(server_names)
        self.capacity = capacity
        self.server_loads = {s: 0 for s in self.server_names}
        self.num_servers = len(self.server_names)
        self.model = create_dlb_model(self.num_servers)
        train_dlb_model(self.model, self.num_servers)
        self.total_predictions = 0
        self.key_assignments = {}

    def assign_key(self, key):
        if not self.server_names:
            return None, 0

        key_feature = hash(key) % 10000
        load_vector = np.array(
            [self.server_loads[s] for s in self.server_names], dtype=float
        )
        input_vector = np.insert(load_vector, 0, key_feature)
        input_tensor = np.expand_dims(input_vector, axis=0)

        pred = self.model.predict(input_tensor, verbose=0)
        idx = int(np.argmax(pred))
        chosen = self.server_names[idx]
        self.total_predictions += 1

        if self.server_loads[chosen] < self.capacity:
            self.server_loads[chosen] += 1
            self.key_assignments[key] = chosen
            return chosen, 0

        return None, 1
