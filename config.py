GPU_MEMORY_GB = 40
MAX_SHARED_JOBS = 4

# Risk thresholds from proposal
SH_THRESHOLD = 0.25
ISO_THRESHOLD = 0.60

TRUST_DOMAINS = ["A", "B", "C", "D"]
SECURITY_LEVELS = [0.25, 0.50, 0.75, 1.00]

# 1 = SH, 2 = ISO, 3 = EX
ISOLATION_LEVELS = [1, 2, 3]
ISOLATION_PROBS = [0.50, 0.35, 0.15]

MODE_STRENGTH = {"SH": 1, "ISO": 2, "EX": 3}
STRENGTH_MODE = {1: "SH", 2: "ISO", 3: "EX"}

DEFAULT_NUM_JOBS = 250
DEFAULT_GPU_COUNTS = [3, 6, 9]
# Finer steps at low load: above ~0.4 jobs/sec every GPU count
# is saturated and the curves flatten out.
DEFAULT_ARRIVAL_RATES = [0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.4]
DEFAULT_SEEDS = list(range(1, 11))
