import torch


print("PyTorch version:", torch.__version__)
print("CUDA available:", torch.cuda.is_available())

if torch.cuda.is_available():
    print("CUDA runtime used by PyTorch:", torch.version.cuda)
    print("GPU:", torch.cuda.get_device_name(0))
    print("GPU count:", torch.cuda.device_count())
else:
    raise SystemExit(
        "CUDA is unavailable. Check the NVIDIA driver and install the correct "
        "PyTorch CUDA wheel from https://pytorch.org/get-started/locally/."
    )
