import torch
import time
import sys

if not torch.backends.mps.is_available():
    print("❌ MPS (Metal Performance Shaders) is not available on this system.")
    sys.exit(1)

print("=========================================================")
print("   PROJECT VALIMELI — MPS GPU MEMORY MONITOR")
print("=========================================================")
print("Press Ctrl+C to stop monitoring.\n")

try:
    while True:
        # Query active allocation from the PyTorch Metal backend (converted to MB)
        allocated = torch.mps.current_allocated_memory() / (1024 * 1024)
        driver = torch.mps.driver_allocated_memory() / (1024 * 1024)
        
        print(f"[{time.strftime('%H:%M:%S')}] "
              f"PyTorch Active: {allocated:.2f} MB | "
              f"Metal Driver Total: {driver:.2f} MB", end="\r", flush=True)
        time.sleep(2)
except KeyboardInterrupt:
    print("\n\nStopped monitoring.")
