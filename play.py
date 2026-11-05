#!/usr/bin/env python
"""SO-101 Twin Depot: `python play.py --help`."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from twin_depot.app import main  # noqa: E402

if __name__ == "__main__":
    main()
