#!/usr/bin/env python3
"""
Occupy GPU 1 and GPU 2 to prevent others from using them.
Press Ctrl+C to release the GPUs.

Usage:
    python occupy_gpus.py
    # or
    CUDA_VISIBLE_DEVICES=1,2 python occupy_gpus.py
"""

import os
import torch
import time
import signal
import sys

# Set which GPUs to occupy (physical GPU 1 and 2)
os.environ["CUDA_VISIBLE_DEVICES"] = "1,2"

print("=" * 80)
print("GPU Occupation Script")
print("=" * 80)
print(f"CUDA_VISIBLE_DEVICES: {os.environ['CUDA_VISIBLE_DEVICES']}")
print(f"Physical GPUs 1,2 will be mapped to logical cuda:0, cuda:1")
print(f"\nPress Ctrl+C to release GPUs and exit")
print("=" * 80)

# Signal handler for graceful exit
def signal_handler(sig, frame):
    print("\n\n" + "=" * 80)
    print("Ctrl+C detected! Releasing GPUs...")
    print("=" * 80)
    # Clear CUDA cache
    if torch.cuda.is_available():
        for i in range(torch.cuda.device_count()):
            with torch.cuda.device(i):
                torch.cuda.empty_cache()
                torch.cuda.synchronize()
    print("GPUs released. Exiting...")
    sys.exit(0)

signal.signal(signal.SIGINT, signal_handler)

# Check CUDA availability
if not torch.cuda.is_available():
    print("ERROR: CUDA is not available!")
    sys.exit(1)

num_gpus = torch.cuda.device_count()
print(f"\nDetected {num_gpus} GPU(s) available:")
for i in range(num_gpus):
    props = torch.cuda.get_device_properties(i)
    print(f"  - Logical GPU {i}: {props.name} ({props.total_memory / 1024**3:.2f} GB)")

# Allocate memory on each GPU (占用约90%的显存，纯占用不计算)
tensors = []
occupation_ratio = 0.90  # Occupy 90% of GPU memory

print(f"\nAllocating {occupation_ratio*100:.0f}% memory on each GPU (idle mode, no computation)...")

for gpu_id in range(num_gpus):
    with torch.cuda.device(gpu_id):
        props = torch.cuda.get_device_properties(gpu_id)
        total_memory = props.total_memory
        # Calculate tensor size (in float32 = 4 bytes)
        tensor_size = int(total_memory * occupation_ratio / 4)
        
        try:
            print(f"  - GPU {gpu_id}: Allocating {tensor_size * 4 / 1024**3:.2f} GB...", end=" ")
            tensor = torch.zeros(tensor_size, dtype=torch.float32, device=f"cuda:{gpu_id}")
            tensors.append(tensor)
            
            # Verify allocation
            allocated = torch.cuda.memory_allocated(gpu_id) / 1024**3
            reserved = torch.cuda.memory_reserved(gpu_id) / 1024**3
            free = (total_memory - torch.cuda.memory_reserved(gpu_id)) / 1024**3
            print(f"✓ (Allocated: {allocated:.2f} GB, Free: {free:.2f} GB)")
        except RuntimeError as e:
            print(f"✗ Failed: {e}")

# Perform some computation to keep GPUs busy
print("\n" + "=" * 80)
print("GPUs are now occupied! (Idle mode - no computation)")
print("Memory allocated but GPUs remain idle to save power")
print("Press Ctrl+C to stop and release GPUs")
print("=" * 80 + "\n")

iteration = 0
start_time = time.time()

try:
    while True:
        iteration += 1
        
        # No computation - just keep the process alive and hold the memory
        # GPUs will remain idle with low power consumption
        
        # Print status every 60 seconds
        if iteration % 600 == 0:
            elapsed = time.time() - start_time
            print(f"[{elapsed:.0f}s] GPUs still occupied | ", end="")
            
            for gpu_id in range(num_gpus):
                allocated = torch.cuda.memory_allocated(gpu_id) / 1024**3
                print(f"GPU{gpu_id}: {allocated:.2f}GB | ", end="")
            print("Status: IDLE (low power)")
        
        # Sleep to reduce CPU usage
        time.sleep(0.1)

except KeyboardInterrupt:
    print("\n\nKeyboard interrupt detected!")
    signal_handler(None, None)
