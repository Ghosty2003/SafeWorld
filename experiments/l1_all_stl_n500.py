"""All grounded STL specifications with CCE policy imagination (500 per split)."""
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from experiments.l1_policy_stl import ALL_GROUNDED, main


if __name__ == "__main__":
    main(default_specs=ALL_GROUNDED, default_n=500)
