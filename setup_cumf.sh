#!/usr/bin/env bash
# setup_cumf.sh — Attempt to build cuMF_als and convert data for comparison.
#
# IMPORTANT CONSTRAINT: cuMF requires F (rank) to be a MULTIPLE OF 10.
# Our K values (16, 32, 48, 64, 96) are NONE of these. So we can only
# compare at cuMF-compatible ranks: 10, 20, 30, 40, 50, 60, 70, 80, 100.
# For the paper, use K=20 (vs our K=16) and K=30 (vs our K=32) as the
# nearest comparable points. The RMSE should be similar across nearby K.
#
# If this script fails (CUDA 11+/12+ API issues are common), use the
# PyTorch ALS baseline instead: python3 baseline_als_gpu.py
#
# Usage: ./setup_cumf.sh /path/to/netflix_ratings.bin [K=20]
set -euo pipefail

BIN_FILE="${1:-/home/pc/Desktop/2007080/netflix_ratings.bin}"
K="${2:-20}"          # Must be a multiple of 10
LAMBDA=0.1
ITERS=50
ARCH="${ARCH:-86}"    # 86 = RTX 3060; 75 = T4
CUMF_DIR="cumf_als"

# ── Prerequisite check ──────────────────────────────────────────────────────
echo "=== cuMF setup for APR-BALS comparison ==="
echo "NOTE: cuMF requires rank (K) to be a multiple of 10."
echo "      Using K=$K. Nearest APR-BALS operating point: see below."
echo ""

if (( K % 10 != 0 )); then
    echo "ERROR: K=$K is not a multiple of 10. cuMF will reject it."
    echo "Valid choices: 10 20 30 40 50 60 70 80 100"
    exit 1
fi

command -v nvcc >/dev/null || { echo "ERROR: nvcc not in PATH"; exit 1; }
CUDA_VER=$(nvcc --version | grep -oP 'release \K[\d.]+' | head -1)
echo "CUDA version: $CUDA_VER"
echo "Target arch : sm_$ARCH"
echo ""

# ── Clone ────────────────────────────────────────────────────────────────────
if [[ ! -d "$CUMF_DIR" ]]; then
    echo "Cloning cuMF_als ..."
    git clone https://github.com/cuMF/cumf_als.git "$CUMF_DIR"
else
    echo "cuMF directory already exists, skipping clone."
fi

# ── Patch Makefile for sm_86 + modern CUDA ──────────────────────────────────
echo "Patching Makefile for sm_$ARCH ..."
cd "$CUMF_DIR"
# Replace default SMS (35) with our target + 35 for backward compat
sed -i "s/SMS ?= 35/SMS ?= 35 $ARCH/" Makefile
# Suppress deprecated warnings that become errors on newer CUDA
sed -i 's/NVCCFLAGS +=/NVCCFLAGS += -Wno-deprecated-gpu-targets /' Makefile || true

echo "Building ..."
if make -j4 2>&1 | tee /tmp/cumf_build.log; then
    echo ""
    echo "=== cuMF build SUCCEEDED ==="
else
    echo ""
    echo "=== cuMF build FAILED ==="
    echo "Common causes:"
    echo "  - CUDA 11+ deprecated some APIs used in cuMF"
    echo "  - Check /tmp/cumf_build.log for details"
    echo ""
    echo "Fallback: use PyTorch ALS baseline instead:"
    echo "  python3 ../baseline_als_gpu.py $BIN_FILE --K 16 --lam $LAMBDA"
    cd ..
    exit 1
fi
cd ..

# ── Convert data ─────────────────────────────────────────────────────────────
echo ""
echo "Converting .bin to cuMF binary COO format ..."
python3 - <<'PYEOF'
import sys, struct, numpy as np, os

bin_path = sys.argv[1] if len(sys.argv) > 1 else "$BIN_FILE"
bin_path = "$BIN_FILE"  # injected from shell

with open(bin_path, "rb") as f:
    version, num_users, num_items, nnz_train, nnz_test = struct.unpack("<iiiii", f.read(20))
    def rv(dtype):
        (n,) = struct.unpack("<Q", f.read(8))
        return np.frombuffer(f.read(n * np.dtype(dtype).itemsize), dtype=dtype, count=n).copy()
    train_u = rv(np.int32); train_i = rv(np.int32); train_r = rv(np.float32)
    test_u  = rv(np.int32); test_i  = rv(np.int32); test_r  = rv(np.float32)

os.makedirs("cumf_data", exist_ok=True)

# cuMF COO binary format: int32 rows/cols, float32 data — three separate files
train_u.astype(np.int32).tofile("cumf_data/train.row")
train_i.astype(np.int32).tofile("cumf_data/train.col")
train_r.astype(np.float32).tofile("cumf_data/train.data")
test_u.astype(np.int32).tofile("cumf_data/test.row")
test_i.astype(np.int32).tofile("cumf_data/test.col")
test_r.astype(np.float32).tofile("cumf_data/test.data")

print(f"Written: cumf_data/ ({num_users} users, {num_items} items)")
print(f"  train: {nnz_train} ratings  test: {nnz_test} ratings")
PYEOF

# ── Run cuMF ─────────────────────────────────────────────────────────────────
echo ""
echo "=== Running cuMF K=$K λ=$LAMBDA ==="
echo "Command: ./cumf_als/main M N F NNZ_TRAIN NNZ_TEST LAMBDA X_BATCH THETA_BATCH DATA_DIR"
echo ""

# Read dimensions from the .bin header
read NUM_USERS NUM_ITEMS NNZ_TRAIN NNZ_TEST < <(python3 - <<'PYEOF'
import struct
with open("$BIN_FILE","rb") as f:
    _, nu, ni, ntr, nte = struct.unpack("<iiiii",f.read(20))
print(nu, ni, ntr, nte)
PYEOF
)

# X_BATCH and THETA_BATCH: number of entities to process per batch
# Large = faster but more VRAM. For K=20, 10000 is safe on 12GB VRAM.
X_BATCH=10000
THETA_BATCH=10000

echo "M=$NUM_USERS N=$NUM_ITEMS F=$K NNZ=$NNZ_TRAIN NNZ_TEST=$NNZ_TEST"
echo ""

time ./cumf_als/main \
    $NUM_USERS $NUM_ITEMS $K $NNZ_TRAIN $NNZ_TEST \
    $LAMBDA $X_BATCH $THETA_BATCH \
    cumf_data/ 2>&1 | tee cumf_k${K}_result.txt

echo ""
echo "=== cuMF done. Output in cumf_k${K}_result.txt ==="
echo ""
echo "To compare RMSE with APR-BALS, search for 'Test RMSE' in the output:"
grep -i "test rmse\|train rmse" cumf_k${K}_result.txt | tail -20 || true
echo ""
echo "APR-BALS comparison points (from main_experiment.cu results):"
echo "  K=16: Test RMSE 0.860656, wall 1.342s"
echo "  K=32: Test RMSE 0.927566, wall 5.120s"
echo "  K=64: Test RMSE 1.074587, wall 16.1s"
echo "  (nearest cuMF K=$K should have similar test RMSE)"
