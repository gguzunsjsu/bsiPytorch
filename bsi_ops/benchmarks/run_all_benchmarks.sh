#!/bin/bash

echo "=== BSI Benchmark Suite ==="
echo

# Initialize and activate conda environment
echo "Initializing conda and activating environment: bsiPytorch..."
echo "Changed the bash"
source /opt/anaconda3/etc/profile.d/conda.sh
conda activate bsiPytorch

# Verify the environment is active
if [ "$CONDA_DEFAULT_ENV" = "bsiPytorch" ]; then
    echo "Successfully activated conda environment: bsiPytorch"
else
    echo "Failed to activate conda environment: bsiPytorch. Exiting."
    exit 1
fi

# Get the directory of this script
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"

# Create results directory
mkdir -p "$SCRIPT_DIR/results"

# Change to benchmarks directory
cd "$SCRIPT_DIR"
pwd

# 1. Calibrate precision factor
echo "1. Calibrating precision factor..."
python calibrate_pf.py
echo

# 2. Run micro benchmarks  
echo "2. Running micro benchmarks..."
python benchmark_bsi_micro.py --pf 31 --normalize
echo

# 3. Run attention-like benchmark
echo "3. Running attention-like benchmark..."
python attention_like_benchmark.py --pf 31
echo

# 4. Run accuracy verification
echo "4. Running accuracy verification on OPT model..."
python verify_accuracy_bsi.py
echo

# 5. Run performance benchmark
echo "5. Running performance benchmark..."
python benchmark_performance_bsi.py
echo

echo "=== BSI Benchmark Suite Complete ==="
echo "Benchmark Suite Complete!"
echo "Results saved in bsi_ops/benchmarks/results/"
echo "================================"