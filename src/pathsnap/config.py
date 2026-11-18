"""Central configuration. Paths can be overridden with environment variables."""
import os

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
DATA_RAW = os.environ.get("PATHSNAP_DATA", os.path.join(ROOT, "data", "raw"))
TRAIN_DIR = os.path.join(DATA_RAW, "train")
STUDENT_TEST_DIR = os.path.join(DATA_RAW, "student_test")
# Evaluator trips are dropped in here on Laptop 1 on the day of evaluation.
EVAL_DIR = os.environ.get("PATHSNAP_EVAL_DIR", os.path.join(ROOT, "data", "eval"))
PROCESSED = os.path.join(ROOT, "data", "processed")
RESULTS = os.path.join(ROOT, "results")
ARTIFACTS = os.path.join(ROOT, "artifacts")

SEED = 42
# Candidate generation (chosen on TRAIN partition only, see results/eda)
CAND_RADIUS_M = 60.0
CAND_K_MAX = 10
# Feature context window (observations either side)
CTX_WIN = 3
# Large-gap threshold (seconds) used by features and the HMM
GAP_S = 10.0
