import sys

import torch


def main():
    print("=" * 46)
    print("PyTorch environment check")
    print("=" * 46)
    print(f"interpreter : {sys.executable}")
    print(f"python      : {sys.version.split()[0]}")
    print(f"torch       : {torch.__version__}")
    print(f"cuda ready  : {torch.cuda.is_available()}")

    if torch.cuda.is_available():
        print(f"gpu         : {torch.cuda.get_device_name(0)}")
        print(f"capability  : {torch.cuda.get_device_capability(0)}")
        print(f"cuda runtime: {torch.version.cuda}")
        print("-" * 46)
        print("matrix multiply on GPU:")
        a = torch.rand(3, 3, device="cuda")
        b = torch.rand(3, 3, device="cuda")
        torch.cuda.synchronize()
        print(a @ b)
        print("-" * 46)
        print("GPU OK. Ready to code.")
    else:
        print("-" * 46)
        print("CUDA not available, CPU only.")
        print("Check that the interpreter points to pytorch_env.")


if __name__ == "__main__":
    main()
