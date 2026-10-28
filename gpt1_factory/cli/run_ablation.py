"""Backward-compatible entry point; now uses the validated experiment factory."""
from .ablation_factory import main

if __name__ == "__main__":
    main()
