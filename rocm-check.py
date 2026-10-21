import torch


def main() -> None:
    print("PyTorch:", torch.__version__)
    print("ROCm/HIP:", torch.version.hip)
    print("GPU Available:", torch.cuda.is_available())
    print("Device:", torch.cuda.get_device_name(0) if torch.cuda.is_available() else "None")


if __name__ == "__main__":
    main()
