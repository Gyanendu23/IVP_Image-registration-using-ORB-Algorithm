import cupy as cp

# Check for available GPUs
try:
    cp.cuda.Device(0).compute_capability  # Replace 0 with the index if you have multiple GPUs
    print("GPU is available and connected!")
except cp.cuda.runtime.CUDARuntimeError as e:
    print("No GPU found or CUDA not properly configured:", e)
