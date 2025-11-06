# config.py

# --- HASHING CONFIGURATION ---
HASH_SPACE_SIZE = 2**32  # Total size of the hash ring
NUM_VIRTUAL_NODES = 100  # Virtual nodes per physical server for better balance
SERVER_CAPACITY = 1000  # Bounded Load capacity for the simulation

# --- SIMULATION CONFIGURATION ---
NUM_SERVERS = 10  # Number of physical servers (bins)
NUM_KEYS_UNIFORM = 8000  # Keys for uniform load test (must be < NUM_SERVERS * CAPACITY)
NUM_KEYS_SKEWED = 8000  # Keys for skewed load test

# --- RJ-CH CONFIGURATION ---
MAX_RJ_ATTEMPTS = 100  # Max jumps before giving up

# --- DLB (DEEP LEARNING) CONFIGURATION ---
DLB_EPOCHS = 5  # Training epochs for the simplified model
DLB_TRAINING_SIZE = 10000  # Number of samples to train the DLB model
DLB_ZIPF_ALPHA = 1.2  # Alpha parameter for Zipf distribution (high skew)
