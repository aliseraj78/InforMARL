import torch
print("PyTorch version:", torch.__version__)
print("CUDA version:", torch.version.cuda)

# Check if CUDA is available
if torch.cuda.is_available():
    print("CUDA is available. PyTorch can use the GPU.")
else:
    print("CUDA is not available. PyTorch cannot use the GPU.")

# Check the CUDA version used by PyTorch
print("PyTorch CUDA version:", torch.version.cuda)

# Print the number of available GPUs
print("Number of GPUs available:", torch.cuda.device_count())

# Print the name of the current GPU
if torch.cuda.is_available():
    print("Current GPU:", torch.cuda.get_device_name(torch.cuda.current_device()))


