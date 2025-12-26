#!/bin/bash

# wait.sh - Wait for GPU availability and run training

echo "Checking for GPU availability..."
echo "Started at: $(date)"

# Function to check if GPU is fully available
check_gpu() {
    if ! command -v nvidia-smi &> /dev/null; then
        echo "ERROR: nvidia-smi not found. CUDA/GPU not available on this system."
        return 1
    fi
    
    # Check if GPU memory is fully free (at least 30GB free for full availability)
    free_memory=$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits | head -1)
    if [ "$free_memory" -lt 30720 ]; then
        return 1
    fi
    
    return 0
}

# Wait for GPU to become available
while ! check_gpu; do
    echo "GPU not fully available. Waiting..."
    sleep 60
done

# GPU is available
echo "GPU is fully available!"
echo "Starting training at: $(date)"

# Change to script directory
cd "$(dirname "$0")"

# Run the training command with nohup
nohup python3 -m src.main > trainingFinal.log 2>&1 &

# Get the process ID
PID=$!
echo "Training started with PID: $PID"
echo "Log file: trainingFinal.log"
echo "To monitor: tail -f trainingFinal.log"

# Save PID to file for easy reference
echo $PID > training.pid
echo "PID saved to training.pid"